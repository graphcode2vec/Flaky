from cProfile import label
import json
import os
import collections
import pandas as pd

def build_map_index():
    df = pd.read_csv("test_results.csv", index_col="Test")
    print(len(df))
    counter = 0
    sum = 0 
    n=set()
    summary_graph = []
    with open("outfolder.txt") as f:
        for l in f.readlines():
            parts = l.split(",")
            src, tgt = parts[0], parts[1]
            class_methods_id_mapping = json.load ( open( os.path.join(src, "class_method_id_mapping.json") ))
            id_string_mappting = collections.defaultdict(list)
            for class_name in class_methods_id_mapping:
                if "$" in class_name:
                    continue
                p=class_name.split("$")
                methods = class_methods_id_mapping[class_name]
                for method_id in methods:
                    method_name = methods[method_id]["name"]
                    id_method_string = p[0] + "#" + method_name
                    if method_name == "<init>":
                        continue
                    if id_method_string not in df.index:
                        n.add(src)
                        continue
                    #if id_method_string == "org.activiti.engine.test.bpmn.usertask.TaskPriorityExtensionsTest#testPriorityExtension":
                    #    print(f"{id_method_string},{parts}, {class_name}, {method_id}")
                    #assert id_method_string not in id_string_mappting["Test"],f"{id_method_string},{parts}, {class_name}, {method_id}"
                    id_string_mappting["Method"].append( method_id )
                    id_string_mappting["Test"].append( id_method_string )
                    id_string_mappting["IsFlaky"].append( df.loc[ id_method_string, "IsFlaky" ] )
                    id_string_mappting["project"].append( src )
                    sum += df.loc[ id_method_string, "IsFlaky" ]
                    counter += 1
            
            mdf = pd.DataFrame.from_dict( id_string_mappting )
            mdf.to_csv(os.path.join(src, "flaky_test.csv"))
            summary_graph.append( mdf )
    print(counter)
    print(sum)
    print(df["IsFlaky"].sum())
    all_graph_info = pd.concat( summary_graph )
    all_graph_info.to_csv("graph_info.csv")
   # print(n)




if __name__ == "__main__":
    build_map_index()

            
            

            


