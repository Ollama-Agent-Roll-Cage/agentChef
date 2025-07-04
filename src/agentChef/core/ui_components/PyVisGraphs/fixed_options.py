import json

def create_options_json(node_size, gravity, spring_strength, layout_type):
    """Create options JSON string for PyVis network"""
    options_dict = {
        "nodes": {
            "shadow": True,
            "scaling": {"min": node_size - 10, "max": node_size + 30},
            "widthConstraint": {"minimum": 100}
        },
        "edges": {
            "smooth": {"type": "cubicBezier", "roundness": 0.5},
            "shadow": True,
            "selectionWidth": 4
        },
        "physics": {
            "barnesHut": {
                "gravitationalConstant": gravity,
                "springLength": 200,
                "springConstant": spring_strength,
                "damping": 0.09,
                "avoidOverlap": 0.5
            },
            "minVelocity": 0.75
        },
        "interaction": {
            "hover": True,
            "navigationButtons": True,
            "multiselect": True,
            "dragNodes": True,
            "zoomView": True
        }
    }
    
    # Apply hierarchical layout if selected
    if layout_type == "hierarchical":
        options_dict["layout"] = {
            "hierarchical": {
                "enabled": True,
                "levelSeparation": 150,
                "nodeSpacing": 100,
                "treeSpacing": 200,
                "direction": "UD",
                "sortMethod": "directed"
            }
        }
        
    return json.dumps(options_dict)
