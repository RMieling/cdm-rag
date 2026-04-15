from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from api.config import AppConfig
from api.utils.logger import rag_logger


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    context: list
    search_query: str
    cypher_query: str


class RAGPipeline:
    """
    LangGraph implementation of Conversational GraphRAG using Neo4j
    """

    def __init__(self, config: AppConfig, db_manager):
        self.config = config
        self.db_manager = db_manager
        self.llm = self._get_llm()

        self.memory = MemorySaver()
        self.graph = self._build_graph()
        self.logger = rag_logger
        self.logger.info("Conversational GraphRAG Pipeline initialized successfully.")

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("contextualize", self.contextualize_node)  # Contextualizes query based on chat history
        workflow.add_node("retrieve", self.retrieve_node)  # Generates and executes Cypher
        workflow.add_node("generate", self.generate_node)  # generates the answer based on the retrieved context

        workflow.add_edge(START, "contextualize")
        workflow.add_edge("contextualize", "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile(checkpointer=self.memory)

    def _get_llm(self):
        provider = self.config.LLM_PROVIDER.lower()
        if provider == "ollama":
            from langchain_ollama import ChatOllama

            return ChatOllama(
                model=self.config.OLLAMA_LLM_MODEL,
                base_url=self.config.OLLAMA_ENDPOINT,
                temperature=self.config.TEMPERATURE,
            )
        elif provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                api_key=self.config.OPENAI_API_KEY,
                model=self.config.OPENAI_MODEL,
                temperature=self.config.TEMPERATURE,
            )
        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")

    def contextualize_node(self, state: AgentState):
        """Rewrite the latest question (based on chat history if available)"""
        messages = state["messages"]
        latest_question = messages[-1].content

        if len(messages) <= 1:
            self.logger.debug(f"DEBUG - No message history exists. Raw Query: {latest_question}")
            return {"search_query": latest_question}

        history_messages = messages[:-1]

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.config.prompts.contextualize_system_prompt),
                MessagesPlaceholder(variable_name="messages"),
                ("human", "{question}"),
            ]
        )

        chain = prompt | self.llm
        response = chain.invoke({"messages": history_messages, "question": latest_question})

        self.logger.debug(f"Rewrote query to: {response.content}")
        return {"search_query": response.content}

    def retrieve_node(self, state: AgentState):
        """Generates a Cypher query, executes it, and tries to self-correct on failure."""
        query = state["search_query"]

        max_retries = 3
        last_error = None
        last_cypher = None

        for attempt in range(max_retries):
            # aggressively pass system prompt back to the LLM on error from previous loop
            messages = [("system", self.config.prompts.cypher_system_prompt)]

            if last_error:
                self.logger.warning(f"Initiating Cypher self-correction (Attempt {attempt + 1}/{max_retries})...")
                correction_prompt = (
                    f"User Question: {query}\n\n"
                    f"Your previous Cypher query:\n{last_cypher}\n\n"
                    f"Failed with the following Neo4j database error:\n{last_error}\n\n"
                    f"Analyze the error, fix the Cypher query, and try again. "
                    f"OUTPUT ONLY THE RAW CYPHER QUERY."
                )
                messages.append(("user", correction_prompt))
            else:
                messages.append(("user", "{question}"))

            cypher_prompt = ChatPromptTemplate.from_messages(messages)
            cypher_chain = cypher_prompt | self.llm

            # Invoke the LLM
            if last_error:
                cypher_response = cypher_chain.invoke({})  # Variables already baked into the string
            else:
                cypher_response = cypher_chain.invoke({"question": query})

            raw_cypher = cypher_response.content.replace("```cypher", "").replace("```", "").strip()

            if raw_cypher == "SKIP_QUERY":
                self.logger.info("Casual conversation detected. Skipping database retrieval.")
                return {"context": [], "cypher_query": "No query needed (Greeting)"}

            self.logger.info(f"Generated Cypher Query (Attempt {attempt + 1}):\n{raw_cypher}")

            # Execute against Neo4j
            records = []
            try:
                with self.db_manager.driver.session() as session:
                    result = session.run(raw_cypher)
                    for record in result:
                        records.append(str(record.data()))

                self.logger.debug(f"Success! Retrieved {len(records)} records from Neo4j.")
                return {"context": records, "cypher_query": raw_cypher}

            except Exception as e:
                # Catch the Neo4j error, save it, and let the loop run again
                self.logger.error(f"Cypher execution failed: {e}")
                last_error = str(e)
                last_cypher = raw_cypher

        # Fallback if all retries are exhausted
        self.logger.critical("Exhausted all Cypher self-correction retries. Moving to generation with error context.")
        return {
            "context": [
                "System Note: Failed to retrieve data from the graph database after multiple attempts. "
                f"The last database error was: {last_error}"
            ],
            "cypher_query": last_cypher,
        }

    def generate_node(self, state: AgentState):
        """Generate answer using the retrieved Neo4j context and chat history"""
        messages = state["messages"]
        context_list = state["context"]

        # Join the stringified JSON records into a single context block
        context_str = "\n".join(context_list) if context_list else "No data found in the graph."

        self.logger.debug(f"Context for generation: {context_str[:500]}...")

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.config.prompts.generate_system_prompt),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        chain = prompt | self.llm
        response = chain.invoke({"messages": messages, "context": context_str})

        return {"messages": [response]}

    def ask(self, question: str, session_id: str = "default_session") -> dict:
        """Executes a query against the LangGraph application."""
        self.logger.info(f"Processing query for thread '{session_id}': {question}")

        try:
            config = {"configurable": {"thread_id": session_id}}
            final_state = self.graph.invoke({"messages": [HumanMessage(content=question)]}, config=config)

            answer = final_state["messages"][-1].content

            # Return the last Cypher query as the "source" so the user might verify the DB was queried!
            cypher_used = final_state.get("cypher_query", "No query generated.")

            return {"answer": answer, "sources": [f"Executed Cypher: {cypher_used}"]}

        except Exception as e:
            self.logger.error(f"Error during LangGraph execution: {e}", exc_info=True)
            return {"answer": "I encountered an error processing the graph data.", "sources": []}
