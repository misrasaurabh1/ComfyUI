def is_link(obj):
    # Fast path checks for link-like objects: [str, (int/float)]
    if type(obj) is list and len(obj) == 2:
        fst, snd = obj
        if type(fst) is str and (type(snd) is int or type(snd) is float):
            return True
    return False

# The GraphBuilder is just a utility class that outputs graphs in the form expected by the ComfyUI back-end
class GraphBuilder:
    _default_prefix_root = ""
    _default_prefix_call_index = 0
    _default_prefix_graph_index = 0

    def __init__(self, prefix = None):
        if prefix is None:
            self.prefix = GraphBuilder.alloc_prefix()
        else:
            self.prefix = prefix
        self.nodes = {}
        self.id_gen = 1

    @classmethod
    def set_default_prefix(cls, prefix_root, call_index, graph_index = 0):
        cls._default_prefix_root = prefix_root
        cls._default_prefix_call_index = call_index
        cls._default_prefix_graph_index = graph_index

    @classmethod
    def alloc_prefix(cls, root=None, call_index=None, graph_index=None):
        if root is None:
            root = GraphBuilder._default_prefix_root
        if call_index is None:
            call_index = GraphBuilder._default_prefix_call_index
        if graph_index is None:
            graph_index = GraphBuilder._default_prefix_graph_index
        result = f"{root}.{call_index}.{graph_index}."
        GraphBuilder._default_prefix_graph_index += 1
        return result

    def node(self, class_type, id=None, **kwargs):
        if id is None:
            id = str(self.id_gen)
            self.id_gen += 1
        id = self.prefix + id
        if id in self.nodes:
            return self.nodes[id]

        node = Node(id, class_type, kwargs)
        self.nodes[id] = node
        return node

    def lookup_node(self, id):
        id = self.prefix + id
        return self.nodes.get(id)

    def finalize(self):
        output = {}
        for node_id, node in self.nodes.items():
            output[node_id] = node.serialize()
        return output

    def replace_node_output(self, node_id, index, new_value):
        node_id = self.prefix + node_id
        to_remove = []
        for node in self.nodes.values():
            for key, value in node.inputs.items():
                if is_link(value) and value[0] == node_id and value[1] == index:
                    if new_value is None:
                        to_remove.append((node, key))
                    else:
                        node.inputs[key] = new_value
        for node, key in to_remove:
            del node.inputs[key]

    def remove_node(self, id):
        id = self.prefix + id
        del self.nodes[id]

class Node:
    def __init__(self, id, class_type, inputs):
        self.id = id
        self.class_type = class_type
        self.inputs = inputs
        self.override_display_id = None

    def out(self, index):
        return [self.id, index]

    def set_input(self, key, value):
        if value is None:
            if key in self.inputs:
                del self.inputs[key]
        else:
            self.inputs[key] = value

    def get_input(self, key):
        return self.inputs.get(key)

    def set_override_display_id(self, override_display_id):
        self.override_display_id = override_display_id

    def serialize(self):
        serialized = {
            "class_type": self.class_type,
            "inputs": self.inputs
        }
        if self.override_display_id is not None:
            serialized["override_display_id"] = self.override_display_id
        return serialized

def add_graph_prefix(graph, outputs, prefix):
    # Change the node IDs and any internal links
    new_graph = {}
    is_link_local = is_link
    pfx = prefix

    for node_id, node_info in graph.items():
        # Compose new id once
        new_node_id = pfx + node_id
        input_dict = node_info.get("inputs")
        if not input_dict:
            # No inputs or inputs is empty
            new_node = {"class_type": node_info["class_type"], "inputs": {}}
        else:
            # Preallocate inputs dict and do in-place assignment
            inputs = {}
            for input_name, input_value in input_dict.items():
                if is_link_local(input_value):
                    # Avoid repeated attribute and item lookups
                    in0 = input_value[0]
                    in1 = input_value[1]
                    inputs[input_name] = [pfx + in0, in1]
                else:
                    inputs[input_name] = input_value
            new_node = {"class_type": node_info["class_type"], "inputs": inputs}
        new_graph[new_node_id] = new_node

    # Preallocate outputs with list comprehension
    append_pfx = pfx
    is_link_outputs = is_link_local
    # This avoids the repeated attribute lookups and in-loop checking; branch into list comp
    new_outputs = [
        [append_pfx + out[0], out[1]] if is_link_outputs(out) else out
        for out in outputs
    ]
    return new_graph, tuple(new_outputs)

