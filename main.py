import networkx as nx
import json
import os
from nspr_model import NSPRFramework

def load_knowledge_graph(data_dir='data'):
    G = nx.DiGraph()
    print(f"Loading data from '{data_dir}' directory...")

    symptoms_path = os.path.join(data_dir, 'symptoms.json')
    diseases_path = os.path.join(data_dir, 'diseases.json')
    doctors_path = os.path.join(data_dir, 'doctors.json')

    if not all(os.path.exists(p) for p in [symptoms_path, diseases_path, doctors_path]):
        raise FileNotFoundError("Data files not found in 'data/' directory.")

    with open(symptoms_path, 'r') as f:
        symptoms = json.load(f)
        for s in symptoms:
            G.add_node(s['id'], type=s['type'])

    with open(diseases_path, 'r') as f:
        diseases = json.load(f)
        for d in diseases:
            G.add_node(d['id'], type='Disease')
            G.add_node(d['required_specialty'], type='Specialty')
            G.add_edge(d['associated_symptom'], d['id'], relation='associated_with')
            G.add_edge(d['id'], d['required_specialty'], relation='requires_specialty')

    with open(doctors_path, 'r') as f:
        doctors = json.load(f)
        for doc in doctors:
            node_id = doc['name']
            G.add_node(node_id, type='Doctor', 
                       fee=doc['fee'], 
                       specialty=doc['specialty'],
                       location_coords=tuple(doc['location']), 
                       insurance=doc['insurance'])
            G.add_edge(doc['specialty'], node_id, relation='has_doctor')

    print(f"Graph constructed: {len(G.nodes)} nodes, {len(G.edges)} edges.")
    return G

def run_scenario(nspr, scenario_name, query, constraints):
    print(f"\n🔹 SCENARIO: {scenario_name}")
    print(f"   Query: '{query}' | Budget: ${constraints['budget']} | Loc: {constraints['location']} | Ins: {constraints['insurance']}")
    
    try:
        recommendations, path_provenance = nspr.recommend(query, constraints)
    except KeyError:
        print(f"   ❌ Error: Symptom '{query}' not found in Knowledge Graph.")
        return

    if not recommendations:
        print("   ❌ No suitable doctors found.")
        return

    print(f"   Found {len(recommendations)} candidates:")
    for i, (doc, score) in enumerate(recommendations):
        print(f"     {i+1}. {doc} (Score: {score:.4f})")
    
    # Explain top result
    top_doc = recommendations[0][0]
    print("\n   📝 EXPLANATION for Top Choice:")
    print("   " + nspr.generate_explanation(top_doc, path_provenance[top_doc], constraints).replace("\n", "\n   "))
    print("-" * 60)

def run_demo():
    try:
        graph = load_knowledge_graph()
    except Exception as e:
        print(e)
        return

    nspr = NSPRFramework(graph)
    
    # --- Define Multiple Test Scenarios ---
    
    # Scenario 1: Orthopedic Issue (Low Budget)
    run_scenario(nspr, 
                 scenario_name="Worker with Back Pain",
                 query="Severe Back Pain",
                 constraints={'budget': 60, 'location': (12, 12), 'insurance': 'Basic'})

    # Scenario 2: Pediatric Issue (High Fever, Premium Insurance)
    run_scenario(nspr, 
                 scenario_name="Child High Fever Emergency",
                 query="High Fever (Child)",
                 constraints={'budget': 200, 'location': (10, 20), 'insurance': 'Premium'})

    # Scenario 3: Cardiology Issue (Chest Pain, Specific Location)
    run_scenario(nspr, 
                 scenario_name="Elderly Chest Pain",
                 query="Chest Pain",
                 constraints={'budget': 150, 'location': (15, 15), 'insurance': 'Gold'})

if __name__ == "__main__":
    run_demo()