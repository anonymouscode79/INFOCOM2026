                                                                   
metadata_dict_apigraph = {
                  10:{
                     'path':"../data/data_processed/api_graph/",
                     'class_ids':[f"{i}" for i in range(84)]+[f"{90+i}" for i in range(84)],
                     'minorityclass_ids':[f"{90+i}" for i in range(84)],
                     'tasks_list':[i for i in range(84)],
                     'task2_list': [f"{i}" for i in range(84)],
                    #  'task_order':[('1','0'),( '3','2'),('5','4' ),( '7','6'),('9' ,'8'),('11','10'),('13','12' ),( '15','14' ),('17','16' ), ('19','18' )]
                    'task_order':[(f'{i}',f'{90+i}') for i in range(84)]
                    # 'task_order':[( '15','14' ),]
                     },
                   
                }
metadata_dict_androzoo= {
                  10:{
                     'path':"../data/data_processed/androzoo/",
                     'class_ids':[f"{i}" for i in range(36)]+[f"{90+i}" for i in range(36)],
                     'minorityclass_ids':[f"{90+i}" for i in range(36)],
                     'tasks_list':[i for i in range(36)],
                     'task2_list': [f"{i}" for i in range(36)],
                    #  'task_order':[('1','0'),( '3','2'),('5','4' ),( '7','6'),('9' ,'8'),('11','10'),('13','12' ),( '15','14' ),('17','16' ), ('19','18' )]
                    'task_order':[(f'{i}',f'{90+i}') for i in range(36)]
                    # 'task_order':[( '15','14' ),]
                     },
                   
                }
metadata_dict_bodmas= {
                  10:{
                     'path':"../data/data_processed/test/bodmas/",
                     'class_ids':[f"{i}" for i in range(12)]+[f"{90+i}" for i in range(12)],
                     'minorityclass_ids':[f"{90+i}" for i in range(12)],
                     'tasks_list':[i for i in range(12)],
                     'task2_list': [f"{i}" for i in range(12)],
                    #  'task_order':[('1','0'),( '3','2'),('5','4' ),( '7','6'),('9' ,'8'),('11','10'),('13','12' ),( '15','14' ),('17','16' ), ('19','18' )]
                    'task_order':[(f'{i}',f'{90+i}') for i in range(12)]
                    # 'task_order':[( '15','14' ),]
                     },
                   
                } 


def initialize_metadata(label):
    metadict = None
    if label == 'androzoo':
        metadict = metadata_dict_androzoo
    elif label == 'api_graph':
        metadict = metadata_dict_apigraph         
    elif label == 'bodmas':
        metadict = metadata_dict_bodmas        
    return metadict       







