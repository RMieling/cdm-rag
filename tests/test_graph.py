import pytest
import os
from neo4j import GraphDatabase

# requires running neo4j instance  
IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

# TODO currently requires manually running neo4j server

# Setup a pytest fixture to handle the database connection
@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Skipping local neo4j graph test.")
@pytest.fixture(scope="module")
def neo4j_driver():
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "neo4j_password"))
    yield driver
    driver.close()

@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Skipping local neo4j graph test.")
def test_financial_account_has_inherited_attribute(neo4j_driver):
    """
    Tests that the FinancialServices Account has access to base commonApplication attributes 
    (like 'creditLimit') by traversing the semantic inheritance edge.
    """
    target_entity_name = "Account"
    target_path_substring = "FinancialServices"
    expected_attribute = "creditLimit" 

    # Notice the [:EXTENDS_CONCEPT|EXTENDS*0..2] part. 
    # It checks the node itself (0 hops), or traverses up to 2 conceptual/standard 
    # inheritance edges to find the base entity.
    query = """
        MATCH (e:Entity {name: $entity_name})
        WHERE e.path CONTAINS $path_substring
        MATCH (e)-[:EXTENDS_CONCEPT*0..2]->(base:Entity)-[:HAS_ATTRIBUTE]->(a:Attribute {name: $attr_name})
        RETURN count(a) > 0 AS has_attribute
    """

    with neo4j_driver.session() as session:
        result = session.run(
            query, 
            entity_name=target_entity_name, 
            path_substring=target_path_substring,
            attr_name=expected_attribute
        )
        
        record = result.single()
        assert record["has_attribute"] is True, \
            f"Failed: {target_entity_name} ({target_path_substring}) cannot reach '{expected_attribute}' via inheritance."

@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Skipping local neo4j graph test.")
def test_financial_account_has_local_attribute(neo4j_driver):
    """
    Tests that the FinancialServices Account still possesses its own 
    local, banking-specific attributes without needing to inherit them.
    """
    # Use a real banking attribute you found when inspecting the UI earlier
    expected_local_attribute = "Tenureyears" 

    query = """
        MATCH (e:Entity {name: 'Account'})-[:HAS_ATTRIBUTE]->(a:Attribute {name: $attr_name})
        WHERE e.path CONTAINS 'FinancialServices'
        RETURN count(a) > 0 AS has_local
    """
    with neo4j_driver.session() as session:
        result = session.run(query, attr_name=expected_local_attribute)
        
        assert result.single()["has_local"] is True, \
            f"Failed: FinancialServices Account is missing its local banking attribute '{expected_local_attribute}'."