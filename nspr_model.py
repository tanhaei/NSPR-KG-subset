import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import math
import random

class NSPRFramework:
    def __init__(self, graph: nx.DiGraph, embedding_dim=128, seed=42):
        """
        Initializes the Neuro-Symbolic Path Reasoning Framework.
        """
        # Set seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.graph = graph
        self.embedding_dim = embedding_dim
        
        # Initialize embeddings (Dynamic based on graph size)
        self.node_embeddings = nn.Embedding(len(graph.nodes), embedding_dim)
        # We have relations: associated_with, requires_specialty, has_doctor
        self.relation_embeddings = nn.Embedding(10, embedding_dim) 
        
        self.node_to_idx = {node: i for i, node in enumerate(graph.nodes)}

    def get_embedding(self, node):
        idx = self.node_to_idx.get(node)
        if idx is None:
            return torch.zeros(self.embedding_dim)
        return self.node_embeddings(torch.tensor(idx))

    def transE_energy(self, path):
        """
        Calculates Semantic Energy (E_sem).
        """
        energy = 0
        for i in range(len(path) - 1):
            h = self.get_embedding(path[i])
            t = self.get_embedding(path[i+1])
            # Simplified relation embedding for demo
            r = self.relation_embeddings(torch.tensor(0)) 
            
            # Score = - || h + r - t ||
            score = -torch.norm(h + r - t, p=2)
            energy += score.item()
        return energy

    def calculate_distance(self, loc1, loc2):
        """
        Calculates Euclidean distance in km (approx).
        """
        x1, y1 = loc1
        x2, y2 = loc2
        # Euclidean distance formula
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2) * 2  # Scale factor for demo km

    def constraint_score(self, doctor_node, user_constraints):
        """
        Calculates Psi (Constraint Satisfaction).
        """
        doc_data = self.graph.nodes[doctor_node]
        
        # 1. Cost Constraint (Sigmoid Decay)
        fee = doc_data.get('fee', 100)
        budget = user_constraints['budget']
        alpha = 0.1 # Adjusted sensitivity
        psi_cost = 1 / (1 + math.exp(alpha * (fee - budget)))
        
        # 2. Geo Constraint (Gaussian Decay)
        user_loc = user_constraints['location']
        doc_loc = doc_data.get('location_coords', (0,0))
        dist = self.calculate_distance(user_loc, doc_loc)
        sigma = 20 # Stricter distance constraint
        psi_geo = math.exp(-(dist**2) / (2 * sigma**2))
        
        # 3. Insurance Constraint (Binary/Fuzzy)
        user_ins = user_constraints['insurance']
        doc_ins = doc_data.get('insurance', [])
        # If user has "None", insurance doesn't matter (score 1.0)
        if user_ins == "None":
            psi_ins = 1.0
        else:
            psi_ins = 1.0 if user_ins in doc_ins else 0.1 
        
        return psi_cost * psi_geo * psi_ins

    def beam_search(self, start_node, target_type='Doctor', k=50, max_depth=4):
        """
        Finds paths from start_node to nodes of target_type.
        Increased beam width (k) to handle larger graph.
        """
        beam = [(start_node, [start_node])]
        valid_paths = []
        
        for _ in range(max_depth):
            candidates = []
            for node, path in beam:
                if self.graph.nodes[node].get('type') == target_type:
                    valid_paths.append(path)
                    continue
                
                neighbors = list(self.graph.neighbors(node))
                for neighbor in neighbors:
                    if neighbor not in path: 
                        candidates.append((neighbor, path + [neighbor]))
            
            random.shuffle(candidates)
            beam = candidates[:k]
        
        for node, path in beam:
            if self.graph.nodes[node].get('type') == target_type:
                valid_paths.append(path)
                
        return valid_paths

    def recommend(self, symptom_node, user_constraints, top_k=3):
        """
        Main reasoning loop.
        """
        paths = self.beam_search(symptom_node, target_type='Doctor')
        
        doctor_scores = {}
        path_provenance = {} 
        
        for path in paths:
            doctor = path[-1]
            e_sem = self.transE_energy(path)
            psi = self.constraint_score(doctor, user_constraints)
            
            semantic_score = math.exp(e_sem) 
            total_score = psi * semantic_score
            
            if doctor not in doctor_scores:
                doctor_scores[doctor] = 0
            doctor_scores[doctor] += total_score
            
            # Save path (overwrite with latest found path for this doc)
            path_provenance[doctor] = path

        ranked_docs = sorted(doctor_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked_docs[:top_k], path_provenance

    def generate_explanation(self, doctor, path, user_constraints):
        """
        Generates explanation string.
        """
        doc_data = self.graph.nodes[doctor]
        symptom = path[0]
        disease = path[1] if len(path) > 1 else "Unknown Condition"
        dist = self.calculate_distance(user_constraints['location'], doc_data['location_coords'])
        
        explanation = (
            f"   ✅ **Recommendation:** {doctor} ({doc_data['specialty']})\n"
            f"   🩺 **Clinical Logic:** User reported '{symptom}' -> Linked to '{disease}' -> Requires {doc_data['specialty']}.\n"
            f"   💰 **Cost Analysis:** Fee is ${doc_data['fee']} (Your Budget: ${user_constraints['budget']}).\n"
            f"   📍 **Location:** ~{dist:.1f} km away.\n"
            f"   yh **Insurance:** {user_constraints['insurance']} "
            f"{'Accepted' if user_constraints['insurance'] in doc_data['insurance'] else '(Not Accepted)'}."
        )
        return explanation