from tree_sitter import Language, Parser
import os
from ollama_documentor import ollama_code_review_generator
import sys
# ======== Build Tree-sitter Language (once) ========
LIB_NAME = 'build/my-languages.so'
JAVA_GRAMMAR_DIR = 'tree-sitter-java'

if not os.path.exists(LIB_NAME):
    Language.build_library(LIB_NAME, [JAVA_GRAMMAR_DIR])

JAVA_LANGUAGE = Language(LIB_NAME, 'java')

parser = Parser()
parser.set_language(JAVA_LANGUAGE)

# ========== Helpers ==========
def parse_java_code(code: str):
    return parser.parse(code.encode()).root_node

def node_text(node, source):
    return source[node.start_byte:node.end_byte]

def extract_by_type(root, type_name):
    stack, result = [root], []
    while stack:
        node = stack.pop()
        if node.type == type_name:
            result.append(node)
        stack.extend(reversed(node.children))
    return result

def extract_identifier_child(node):
    for child in node.children:
        if child.type == "identifier":
            return child
    return None

def extract_type_child(node):
    for child in node.children:
        if child.type == "type":
            return child
    return None

# ========== Extraction Functions ==========
def extract_classes(code, root):
    classes = []
    for node in extract_by_type(root, "class_declaration"):
        name_node = extract_identifier_child(node)
        classes.append({
            "name": node_text(name_node, code) if name_node else None,
            "node": node
        })
    return classes

def extract_methods(code, root):
    methods = []
    for node in extract_by_type(root, "method_declaration"):
        name_node = extract_identifier_child(node)
        return_type = extract_type_child(node)
        params = extract_by_type(node, "formal_parameter")
        method_params = [{
            "name": node_text(p.child_by_field_name("name"), code),
            "type": node_text(p.child_by_field_name("type"), code)
        } for p in params]
        methods.append({
            "name": node_text(name_node, code),
            "return_type": node_text(return_type, code) if return_type else None,
            "parameters": method_params,
            "node": node
        })
    return methods

def extract_fields(code, root):
    fields = []
    for node in extract_by_type(root, "field_declaration"):
        type_node = extract_type_child(node)
        var_node = extract_by_type(node, "variable_declarator")[0]
        fields.append({
            "name": node_text(var_node.child_by_field_name("name"), code),
            "type": node_text(type_node, code) if type_node else None
        })
    return fields

def extract_constructors(code, root):
    constructors = []
    for node in extract_by_type(root, "constructor_declaration"):
        name_node = extract_identifier_child(node)
        params = extract_by_type(node, "formal_parameter")
        method_params = [{
            "name": node_text(p.child_by_field_name("name"), code),
            "type": node_text(p.child_by_field_name("type"), code)
        } for p in params]
        constructors.append({
            "name": node_text(name_node, code),
            "parameters": method_params
        })
    return constructors

def extract_annotations(code, root):
    annotations = []
    for node in extract_by_type(root, "marker_annotation") + extract_by_type(root, "annotation"):
        annotations.append({
            "text": node_text(node, code),
            "node": node
        })
    return annotations

def extract_imports(code, root):
    return [node_text(node, code) for node in extract_by_type(root, "import_declaration")]

def extract_enums(code, root):
    enums = []
    for node in extract_by_type(root, "enum_declaration"):
        name_node = extract_identifier_child(node)
        enums.append({
            "name": node_text(name_node, code),
            "node": node
        })
    return enums

def extract_interfaces(code, root):
    interfaces = []
    for node in extract_by_type(root, "interface_declaration"):
        name_node = extract_identifier_child(node)
        interfaces.append({
            "name": node_text(name_node, code),
            "node": node
        })
    return interfaces

# ========== Master Function ==========
def extract_java_elements(code: str):
    root = parse_java_code(code)
    return {
        "classes": extract_classes(code, root),
        "methods": extract_methods(code, root),
        "constructors": extract_constructors(code, root),
        "fields": extract_fields(code, root),
        "annotations": extract_annotations(code, root),
        "imports": extract_imports(code, root),
        "enums": extract_enums(code, root),
        "interfaces": extract_interfaces(code, root)
    }


if __name__ == "__main__":
    f = open("talend_job.java", "r", encoding="utf-8")
    prompt = """Act like you are an expert in java programming language, 
                you need to reverse engineer the business rules from the given code snippet.
                write a detailed document on the given method so that it can be used by another model to 
                generate the same code.return only the required tokens so that it can be used to prompt,providing the code snippet
                \n{}"""
    reception_promt = """Act like you are an expert in python  programming language, 
                         you need to reverse engineer python code from the business rules and tokens below
                         only return the converted code and nothing else. no commentary or extra messages, only code.
                         \n{}"""
    sample_code = f.read()
    ollama_send = ollama_code_review_generator(prompt_template = prompt)
    ollama_recieve = ollama_code_review_generator(prompt_template = reception_promt)
    elements = extract_java_elements(sample_code)
    for k, v in elements.items():
        if k.lower() == "methods":
            print(f"\n=== {k.upper()} ===")
            for item in v:
                # print(item if isinstance(item, str) else item.get("name"))
                # print(sample_code[item.get('node').start_byte:item.get('node').end_byte])
                name = item if isinstance(item, str) else item.get("name")
                print(name)

                # Get actual code snippet using Tree-sitter node
                node = item.get('node')
                snippet = sample_code[node.start_byte:node.end_byte]

                # Send and print Ollama response
                print("\n".join(ollama_recieve("\n".join(ollama_send(snippet)))))
                    