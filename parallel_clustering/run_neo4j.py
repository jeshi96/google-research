import os
import stellargraph as sg
from stellargraph.connector.neo4j import Neo4jStellarGraph
from stellargraph.layer import GCN
from stellargraph.mapper import ClusterNodeGenerator
import tensorflow as tf
import py2neo
import os
from sklearn import preprocessing, feature_extraction, model_selection
import time

import numpy as np
import scipy.sparse as sps
import pandas as pd


def main():
  read_dir = "/home/jeshi/snap/"
  write_dir = "/home/jeshi/neo4j/"
  files = ["amazon.edges","dblp.edges","lj.edges","orkut.edges"]
  filename = read_dir + sys.argv[1]
  edge_list = pd.read_csv(
    os.path.join(read_dir, filename),
    sep="\t",
    header=None,
    names=["target", "source"],
  )
  max_values = max(edge_list["target"].max(), edge_list["source"].max())
  nodes = [x for x in range(max_values + 1)]
  node_list = pd.DataFrame(np.asarray(nodes), columns=["id"])

  default_host = os.environ.get("STELLARGRAPH_NEO4J_HOST")
  graph = py2neo.Graph(host=default_host, port=None, user=None, password=None)

  empty_db_query = """
    MATCH(n) DETACH
    DELETE(n)
    """

  tx = graph.begin(autocommit=True)
  tx.evaluate(empty_db_query)

  constraints = graph.run("CALL db.constraints").data()
  for constraint in constraints:
    graph.run(f"DROP CONSTRAINT {constraint['name']}")

  indexes = graph.run("CALL db.indexes").data()
  for index in indexes:
    graph.run(f"DROP INDEX {index['name']}")

  loading_node_query = """
    UNWIND $node_list as node
    CREATE( e: paper {
      ID: toInteger(node.id)
    })
    """

  # For efficient loading, we will load batch of nodes into Neo4j.
  batch_len = 500

  for batch_start in range(0, len(node_list), batch_len):
    batch_end = batch_start + batch_len
    # turn node dataframe into a list of records
    records = node_list.iloc[batch_start:batch_end].to_dict("records")
    tx = graph.begin(autocommit=True)
    tx.evaluate(loading_node_query, parameters={"node_list": records})

  loading_edge_query = """
    UNWIND $edge_list as edge

    MATCH(source: paper {ID: toInteger(edge.source)})
    MATCH(target: paper {ID: toInteger(edge.target)})

    MERGE (source)-[r:cites]->(target)
    """

  batch_len = 500

  for batch_start in range(0, len(edge_list), batch_len):
    batch_end = batch_start + batch_len
    # turn edge dataframe into a list of records
    records = edge_list.iloc[batch_start:batch_end].to_dict("records")
    tx = graph.begin(autocommit=True)
    tx.evaluate(loading_edge_query, parameters={"edge_list": records})

  node_id_constraint = """
    CREATE CONSTRAINT
    ON (n:paper)
    ASSERT n.ID IS UNIQUE
    """

  tx = graph.begin(autocommit=True)
  tx.evaluate(node_id_constraint)

  neo4j_sg = Neo4jStellarGraph(graph)

  start = time.time()
  clusters = neo4j_sg.clusters(method="louvain")
  end = time.time()
  print("Time: " + str(end-start))

  output_fp = write_dir + filename + ".out"
  with open(output_fp, 'w') as fp:
    for plm in clusters:
      for p in plm:
        print(str(p),end='\t',flush=True,file=fp)
      print(flush=True,file=fp)

if __name__ == "__main__":
  main()