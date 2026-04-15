import asyncio
from typing import Dict

from cdm.objectmodel import CdmEntityDefinition, CdmManifestDefinition
from neo4j import GraphDatabase

from api.config import get_config
from api.services.parse_cdm import fetch_and_traverse_manifest
from api.utils.logger import ingestion_logger


class Neo4jGraphManager:
    """
    Manages the lifecycle of the Neo4j Graph Database, including connection management,
    idempotent ingestion of CDM schemas, and post-processing semantic links.
    """

    def __init__(self, config):
        self.config = config
        self.logger = ingestion_logger

        self.uri = self.config.NEO4J_URI
        self.driver = GraphDatabase.driver(self.uri, auth=(self.config.NEO4J_USERNAME, self.config.NEO4J_PASSWORD))
        self.logger.info(f"Neo4jGraphManager initialized and connected to {self.uri}")

    def close(self):
        """Cleanly shuts down the Neo4j driver connection."""
        if self.driver:
            self.driver.close()
            self.logger.info("Neo4j driver connection closed.")

    @staticmethod
    def _ingest_entities(tx, resolved_entity: CdmEntityDefinition):
        """Translates a resolved CDM entity into Graph Nodes and Attributes."""
        entity_name = resolved_entity.entity_name.replace("_Resolved", "")
        entity_description = resolved_entity.description
        doc = resolved_entity.in_document

        # Fixed string quotes inside the f-string for Python compatibility
        entity_path = f"{doc.folder_path}{doc.name.replace('_Resolved', '')}"

        if not entity_path:
            print(f"Source name not defined for resolved entity: {resolved_entity.entity_name}. Skipping..")
            return

        # 1. Merge Entity
        tx.run(
            """
            MERGE (e:Entity {path: $entity_path})
            SET e.name = $entity_name
            SET e.description = $entity_description
        """,
            entity_path=entity_path,
            entity_name=entity_name,
            entity_description=entity_description,
        )

        # 2. Merge Attributes
        for attr in resolved_entity.attributes:
            tx.run(
                """
                MATCH (e:Entity {path: $entity_path})
                MERGE (a:Attribute {name: $attr_name, entity_path: $entity_path})
                SET a.source_name = $source_name,
                    a.display_name = $display_name,
                    a.description = $description
                MERGE (e)-[:HAS_ATTRIBUTE]->(a)
            """,
                entity_path=entity_path,
                attr_name=attr.name,
                source_name=attr.source_name,
                display_name=attr.display_name,
                description=attr.description or "",
            )

    @staticmethod
    def _ingest_manifest_relationships(tx, manifest: CdmManifestDefinition):
        """Translates CDM manifest relationships into Graph Edges."""
        if not manifest or not manifest.relationships:
            return

        filter_for_now = ["foundationCommon", "wellKnownCDSAttributeGroups"]

        for rel in manifest.relationships:
            # Noise filter
            if any(noise in rel.from_entity for noise in filter_for_now) or any(
                noise in rel.to_entity for noise in filter_for_now
            ):
                continue  # Use continue instead of return to process the rest of the list

            tx.run(
                """
                MERGE (from:Entity {name: $from_entity})
                MERGE (to:Entity {name: $to_entity})
                MERGE (from)-[r:RELATES_TO]->(to)
                SET r.from_attribute = $from_attr,
                    r.to_attribute = $to_attr
            """,
                from_entity=rel.from_entity,
                to_entity=rel.to_entity,
                from_attr=rel.from_entity_attribute,
                to_attr=rel.to_entity_attribute,
            )

    @staticmethod
    def _link_semantic_concepts(tx):
        """
        Finds disjointed entities with the exact same name across different
        namespaces (e.g., FinancialServices Account vs applicationCommon Account) and links them.
        """
        tx.run("""
            // Find the "Base" entities in the applicationCommon folder
            MATCH (core:Entity)
            WHERE core.path CONTAINS 'applicationCommon'

            // Find "Industry" entities that have the exact same name
            MATCH (industry:Entity)
            WHERE industry.name = core.name
              AND industry.path <> core.path

            // Draw a semantic link so the LLM knows they are the same concept
            MERGE (industry)-[r:EXTENDS_CONCEPT]->(core)
        """)

    async def ingest_manifests(self, traverse_manifest_files: Dict[str, bool], load_cached_resolved: bool = True):
        """
        The main orchestration pipeline. Acts similar to `ingest_files` in the VectorStoreManager.
        Traverses the target manifests, triggers idempotent ingestion, and links semantics.
        """
        already_ingested = self.get_ingested_manifests()
        new_manifests = {
            path: traverse for path, traverse in traverse_manifest_files.items() if path not in already_ingested
        }

        if not new_manifests:
            self.logger.info(
                f"All {len(traverse_manifest_files)} target manifests are already in the database. Skipping ingestion."
            )
            return

        self.logger.info(
            f"Starting ingestion for {len(new_manifests)} new manifests out of {len(traverse_manifest_files)} total."
        )

        # Define the synchronous closures that will act as callbacks for the parser
        def handle_node_ingestion(resolved_entity: CdmEntityDefinition):
            with self.driver.session() as session:
                session.execute_write(self._ingest_entities, resolved_entity)

        def handle_manifest_ingestion(manifest: CdmManifestDefinition):
            with self.driver.session() as session:
                session.execute_write(self._ingest_manifest_relationships, manifest)

        # Parse and Ingest Nodes & Manifest Relationships
        successfully_ingested = []
        for manifest_path, traverse_submanifests in new_manifests.items():
            self.logger.info(f"Traversing Manifest: {manifest_path} | Sub-manifests: {traverse_submanifests}")
            try:
                await fetch_and_traverse_manifest(
                    manifest_name=str(manifest_path),
                    load_cached_resolved=load_cached_resolved,
                    parent_manifest=None,
                    traverse_submanifests=traverse_submanifests,
                    show_progress=False,
                    on_node_resolved=handle_node_ingestion,
                    on_manifest_parsed=handle_manifest_ingestion,
                )
                successfully_ingested.append(manifest_path)
                self.logger.info(f"Successfully processed manifest: {manifest_path}")
            except Exception as e:
                self.logger.error(f"An error occurred during traversal of {manifest_path}: {e}")

        # Record all successfully ingested manifests in Neo4j
        self.logger.info(f"Marking {len(successfully_ingested)} manifests as ingested...")
        try:
            with self.driver.session() as session:
                for manifest_path in successfully_ingested:
                    session.execute_write(self._mark_manifest_ingested, manifest_path)
        except Exception as e:
            self.logger.error(f"Failed to record ingestion tracking: {e}")

        # Post-Processing: Link Industry entities to Common entities
        self.logger.info("Building semantic bridges between Industry and Core concepts...")
        try:
            with self.driver.session() as session:
                session.execute_write(self._link_semantic_concepts)
        except Exception as e:
            self.logger.error(f"Failed to link semantic concepts: {e}")

        self.logger.info("Graph Database ingestion pipeline complete!")

    def get_ingested_manifests(self) -> set:
        """Queries the graph database to find which manifests have already been processed."""
        try:
            query = "MATCH (t:IngestionTracker) RETURN t.manifest_path AS path"
            with self.driver.session() as session:
                result = session.run(query)
                ingested_paths = {record["path"] for record in result}

            self.logger.debug(f"Found {len(ingested_paths)} previously ingested manifests.")
            return ingested_paths
        except Exception as e:
            self.logger.error(f"Error checking existing ingestion trackers: {e}")
            return set()

    def _mark_manifest_ingested(self, tx, manifest_path: str):
        """Creates a tracker node to record that this manifest is fully ingested."""
        tx.run(
            """
            MERGE (t:IngestionTracker {manifest_path: $path})
            SET t.last_updated = timestamp(),
                t.status = 'COMPLETED'
        """,
            path=manifest_path,
        )


if __name__ == "__main__":
    config = get_config()
    config.NEO4J_URI = "bolt://localhost:7687"
    db_manager = Neo4jGraphManager(config=config)

    # Define the targeted manifests to ingest
    traverse_manifest_files = {
        "/core/applicationCommon/applicationCommon.manifest.cdm.json": False,
        "/core/operationsCommon/Entities/Common/Common.manifest.cdm.json": True,
        "/core/operationsCommon/Entities/Finance/Finance.manifest.cdm.json": True,
        "/FinancialServices/FinancialServices.manifest.cdm.json": True,
    }

    # Trigger Ingestion
    asyncio.run(db_manager.ingest_manifests(traverse_manifest_files=traverse_manifest_files, load_cached_resolved=True))

# # --- GRAPH INGESTION LOGIC (Nodes) ---
# def ingest_entities(tx, resolved_entity: CdmEntityDefinition):
#     entity_name = resolved_entity.entity_name.replace("_Resolved","")

#     doc = resolved_entity.in_document
#     entity_path = f"{doc.folder_path}{doc.name.replace("_Resolved","")}"

#     if not entity_path:
#         print(f"Source name not defined for resolved enitity: {resolved_entity}. Skipping..")
#         return

#     tx.run("""
#         MERGE (e:Entity {path: $entity_path})
#         SET e.name = $entity_name
#     """, entity_path=entity_path, entity_name=entity_name)

#     for attr in resolved_entity.attributes:
#         tx.run("""
#             MATCH (e:Entity {path: $entity_path})
#             MERGE (a:Attribute {name: $attr_name, entity_path: $entity_path})
#             SET a.source_name = $source_name,
#                 a.display_name = $display_name,
#                 a.description = $description
#             MERGE (e)-[:HAS_ATTRIBUTE]->(a)
#         """,
#         entity_path=entity_path,
#         attr_name=attr.name,
#         source_name=attr.source_name,
#         display_name=attr.display_name,
#         description=attr.description or "")

# # --- GRAPH INGESTION LOGIC (Edges) ---
# def ingest_manifest_relationships(tx, manifest: CdmManifestDefinition):
#     if not manifest or not manifest.relationships:
#         return

#     filter_for_now = [
#         "foundationCommon",
#         "wellKnownCDSAttributeGroups"
#     ]

#     for rel in manifest.relationships:

#         if any(noise in rel.from_entity for noise in filter_for_now) or \
#             any(noise in rel.to_entity for noise in filter_for_now):
#             return

#         tx.run("""
#             MERGE (from:Entity {name: $from_entity})
#             MERGE (to:Entity {name: $to_entity})
#             MERGE (from)-[r:RELATES_TO]->(to)
#             SET r.from_attribute = $from_attr,
#                 r.to_attribute = $to_attr
#         """,
#         from_entity=rel.from_entity, to_entity=rel.to_entity,
#         from_attr=rel.from_entity_attribute, to_attr=rel.to_entity_attribute)


# def link_semantic_concepts(tx):
#     """
#     Finds disjointed entities with the exact same name across different
#     namespaces (e.g., Banking Account vs CRM Account) and links them.
#     """
#     tx.run("""
#         // Find the "Base" entities in the applicationCommon folder
#         MATCH (core:Entity)
#         WHERE core.path CONTAINS 'applicationCommon'

#         // Find "Industry" entities that have the exact same name
#         MATCH (industry:Entity)
#         WHERE industry.name = core.name
#           AND industry.path <> core.path

#         // Draw a semantic link so the LLM knows they are the same concept
#         MERGE (industry)-[r:EXTENDS_CONCEPT]->(core)
#     """)


# # --- MAIN EXECUTION ---
# async def traverse_and_ingest_manifest_files(traverse_manifest_files: dict):
#     # Initialize the Neo4j driver
#     config = get_config()

#     driver = GraphDatabase.driver("bolt://localhost:7687", auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD))

#     # Define the synchronous closures that will act as callbacks
#     def handle_node_ingestion(resolved_entity: CdmEntityDefinition):
#         with driver.session() as session:
#             session.execute_write(ingest_entities, resolved_entity)

#     def handle_manifest_ingestion(manifest: CdmManifestDefinition):
#         with driver.session() as session:
#             session.execute_write(ingest_manifest_relationships, manifest)

#     load_cached_resolved = True

#     for manifest_path, traverse_submanifests in traverse_manifest_files.items():
#         print(f"\n--- Traversing Manifest: {manifest_path} ---")
#         try:
#             await fetch_and_traverse_manifest(
#                 manifest_name=str(manifest_path),
#                 load_cached_resolved=load_cached_resolved,
#                 parent_manifest=None,
#                 traverse_submanifests=traverse_submanifests,
#                 show_progress=False,
#                 # Pass the callbacks into the engine
#                 on_node_resolved=handle_node_ingestion,
#                 on_manifest_parsed=handle_manifest_ingestion
#             )
#         except Exception as e:
#             print(f"An error occurred during traversal: {e}")

#     # try to link the industry derived entities to the common ones through link_semantic_concepts
#     with driver.session() as session:
#         session.execute_write(link_semantic_concepts)

#     print("\nIngestion complete!")
#     driver.close()

# if __name__ == "__main__":
#     # Define your targets
#     traverse_manifest_files = {
#         "/core/applicationCommon/applicationCommon.manifest.cdm.json": False,
#         "/core/operationsCommon/Entities/Common/Common.manifest.cdm.json": True,
#         "/core/operationsCommon/Entities/Finance/Finance.manifest.cdm.json": True,
#         "/FinancialServices/FinancialServices.manifest.cdm.json": True,
#     }

#     asyncio.run(traverse_and_ingest_manifest_files(traverse_manifest_files))
