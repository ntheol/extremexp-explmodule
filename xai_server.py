import grpc
from concurrent import futures
import xai_service_pb2_grpc
import xai_service_pb2
from xai_service_pb2_grpc import ExplanationsServicer
import json
import numpy as np
from modules.lib_IF import preprocess_data
from concurrent import futures
from modules.lib_IF import *
# import torch.nn.functional as F
import json
from sklearn.inspection import partial_dependence
from modules.lib import *
from modules.ale import *
from modules.explainers import *
import dice_ml
from modules.ALE_generic import ale
import joblib
import io
from PyALE import ale
import ast 
from aix360.algorithms.protodash import ProtodashExplainer
import dill as pickle
from ExplainabilityMethodsRepository.ExplanationsHandler import *

class ExplainabilityExecutor(ExplanationsServicer):

    def GetExplanation(self, request, context):
        print('Reading data')
        models = json.load(open("metadata/models.json"))
        data = json.load(open("metadata/datasets.json"))
        dataframe = pd.DataFrame()
        label = pd.DataFrame()

        #for request in request_iterator:
        explanation_type = request.explanation_type
        explanation_method = request.explanation_method
        model_name = request.model

        dispatch_table = {
            (explanation_type, 'pdp'): PDPHandler(),
            (explanation_type, '2dpdp'): TwoDPDPHandler(),
            (explanation_type, 'ale'): ALEHandler(),
            (explanation_type, 'counterfactuals'): CounterfactualsHandler(),
            ('featureExplanation', 'prototypes'): PrototypesHandler()
            # Add more handlers as needed
        }
        
        handler = dispatch_table.get((explanation_type, explanation_method))

        if handler:
            return handler.handle(request, models, data, model_name, explanation_type)
        else:
            raise ValueError(f"Unsupported explanation method '{explanation_method}' for type '{explanation_type}'")

    def Initialization(self, request, context):
        models = json.load(open("metadata/models.json"))
        data = json.load(open("metadata/datasets.json"))
        model_name = request.model_name
        model_id = request.model_id

        dispatch_table = {
            ('hyperparameterExplanation', 'pdp'): PDPHandler(),
            ('hyperparameterExplanation', '2dpdp'): TwoDPDPHandler(),
            ('hyperparameterExplanation', 'ale'): ALEHandler(),
            ('hyperparameterExplanation', 'counterfactuals'): CounterfactualsHandler(),
            ('featureExplanation', 'pdp'): PDPHandler(),
            ('featureExplanation', '2dpdp'): TwoDPDPHandler(),
            ('featureExplanation', 'ale'): ALEHandler(),
            ('featureExplanation', 'counterfactuals'): CounterfactualsHandler(),
            ('featureExplanation', 'prototypes'): PrototypesHandler()
        }
            # Add more handlers as needed
        
      #  Load trained model if exists
        try:
            with open(models[model_name]['original_model'], 'rb') as f:
                original_model = joblib.load(f)
        except FileNotFoundError:
            print("Model does not exist. Load existing model.")

        # Load Data
        train = pd.read_csv(data[model_name]['train'],index_col=0) 
        train_labels = pd.read_csv(data[model_name]['train_labels'],index_col=0) 
        test = pd.read_csv(data[model_name]['test'],index_col=0) 
        test_labels = pd.read_csv(data[model_name]['test_labels'],index_col=0) 
        test['label'] = test_labels
        # dataframe = pd.concat([train.reset_index(drop=True), train_labels.reset_index(drop=True)], axis = 1)

        # predictions = original_model.predict(test)
        # test['Predicted'] = predictions
        # test['Label'] = (test['label'] != test['Predicted']).astype(int)

        # missclassified_instances = test[test['Label']==1]

        # param_grid = original_model.param_grid
        # param_grid = transform_grid_plt(param_grid)
        
        # # Load surrogate models for PDP - ALE if exists
        # try:
        #     with open(models[model_id]['pdp_ale_surrogate_model'], 'rb') as f:
        #         pdp_ale_surrogate_model = joblib.load(f)
        # except FileNotFoundError:
        #     print("Surrogate model does not exist. Training new surrogate model") 
        #     pdp_ale_surrogate_model = proxy_model(param_grid,original_model,'accuracy','XGBoostRegressor')
        #     joblib.dump(pdp_ale_surrogate_model, models[model_id]['pdp_ale_surrogate_model'])  

        # # Load surrogate model for CF if exists
        # try:
        #     with open(models[model_id]['cfs_surrogate_model'], 'rb') as f:
        #         cfs_surrogate_model = joblib.load(f)
        #         proxy_dataset = pd.read_csv(models[model_id]['cfs_surrogate_dataset'],index_col=0)
        # except FileNotFoundError:
        #     print("Surrogate model does not exist. Training new surrogate model") 
        #     train = pd.read_csv(data[model_id]['train'],index_col=0) 
        #     train_labels = pd.read_csv(data[model_id]['train_labels'],index_col=0) 
        #     cfs_surrogate_model , proxy_dataset = instance_proxy(train,train_labels,original_model, query,original_model.param_grid)
        #     joblib.dump(cfs_surrogate_model, models[model_id]['cfs_surrogate_model'])  
        #     proxy_dataset.to_csv(models[model_id]['cfs_surrogate_dataset'])

        # ---------------------- Run Explainability Methods for Pipeline -----------------------------------------------

        #PDP
        # x,y = ComputePDP(param_grid=param_grid, model=pdp_ale_surrogate_model, feature=list(param_grid.keys())[0])
        # # 2D PDP
        # x2d,y2d,z = ComputePDP2D(param_grid=param_grid, model=pdp_ale_surrogate_model,feature1=list(param_grid.keys())[0],feature2=list(param_grid.keys())[1])
        # # ALE
        # ale_eff_hp = ComputeALE(param_grid=param_grid, model=pdp_ale_surrogate_model, feature=list(param_grid.keys())[0])

        # #Counterfactuals
        # param_grid = transform_grid(original_model.param_grid)
        # param_space, name = dimensions_aslists(param_grid)
        # space = Space(param_space)

        # plot_dims = []
        # for row in range(space.n_dims):
        #     if space.dimensions[row].is_constant:
        #         continue
        #     plot_dims.append((row, space.dimensions[row]))
        # iscat = [isinstance(dim[1], Categorical) for dim in plot_dims]
        # categorical = [name[i] for i,value in enumerate(iscat) if value == True]
        # proxy_dataset[categorical] = proxy_dataset[categorical].astype(str)
        # query = pd.DataFrame.from_dict(original_model.best_params_,orient='index').T
        # query[categorical] = query[categorical].astype(str)

        # d = dice_ml.Data(dataframe=proxy_dataset, 
        #     continuous_features=proxy_dataset.drop(columns='BinaryLabel').select_dtypes(include='number').columns.tolist()
        #     , outcome_name='BinaryLabel')
        
        # # Using sklearn backend
        # m = dice_ml.Model(model=cfs_surrogate_model, backend="sklearn")
        # # Using method=random for generating CFs
        # exp = dice_ml.Dice(d, m, method="random")
        # e1 = exp.generate_counterfactuals(query, total_CFs=5, desired_class="opposite",sample_size=5000)

        # cfs = e1.cf_examples_list[0].final_cfs_df
        # dtypes_dict = proxy_dataset.drop(columns='BinaryLabel').dtypes.to_dict()
        # for col, dtype in dtypes_dict.items():
        #     cfs[col] = cfs[col].astype(dtype)

        # scaled_query, scaled_cfs = min_max_scale(proxy_dataset=proxy_dataset,factual=query.copy(deep=True),counterfactuals=cfs.copy(deep=True))
        # cfs['Cost'] = cf_difference(scaled_query, scaled_cfs)
        # cfs = cfs.sort_values(by='Cost')
        # query['BinaryLabel'] = 1
        # query['Cost'] = '-'
        # cfs['Type'] = 'Counterfactual'
        # query['Type'] = 'Factual'
        # # for col in query.columns:
        # #     cfs[col] = cfs[col].apply(lambda x: '-' if x == query.iloc[0][col] else x)
        # cfs = pd.concat([query,cfs])

        # ---------------------- Run Explainability Methods for Model -----------------------------------------------

        # PD Plots
        # numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        # features = train.columns.tolist()[0]
        # numeric_features = train.select_dtypes(include=numerics).columns.tolist()
        # categorical_features = train.columns.drop(numeric_features)

        # pdp = partial_dependence(original_model, train, features = [train.columns.tolist().index(features)],
        #                         feature_names=train.columns.tolist(),categorical_features=categorical_features)

        # pdp_grid = [value.tolist() for value in pdp['grid_values']][0]
        # pdp_vals = [value.tolist() for value in pdp['average']][0]

        # #ALE Plots
        # if train[features].dtype in ['int','float']:
        #     ale_eff_feat = ale(X=train, model=original_model, feature=[features],plot=False, grid_size=50, include_CI=True, C=0.95)
        # else:
        #     ale_eff_feat = ale(X=train, model=original_model, feature=[features],plot=False, grid_size=50, predictors=train.columns.tolist(), include_CI=True, C=0.95)

        # CounterFactuals

        # d = dice_ml.Data(dataframe=dataframe, 
        # continuous_features=train.select_dtypes(include='number').columns.tolist()
        # , outcome_name='label')

        # # Using sklearn backend
        # m = dice_ml.Model(model=original_model, backend="sklearn")
        # # Using method=random for generating CFs
        # exp = dice_ml.Dice(d, m, method="random")
        # e1 = exp.generate_counterfactuals(missclassified_instances.reset_index(drop=True).loc[0].to_frame().T.drop(columns=['Predicted','label','Label']), total_CFs=5, desired_class="opposite",sample_size=5000)
        # e1.visualize_as_dataframe(show_only_changes=True)
        # cfs_feat = e1.cf_examples_list[0].final_cfs_df
        # # cfs_feat = pd.concat([missclassified_instances.reset_index(drop=True).loc[0].to_frame().T.drop(columns=['label','Label']),cfs_feat])
        # query_feat = missclassified_instances.reset_index(drop=True).loc[0].to_frame().T
        # query_feat = query_feat.drop(columns=['label','Label'])
        # query_feat.rename(columns={'Predicted':'label'},inplace=True)
        # # for col in query_feat.columns:
        # #     cfs_feat[col] = cfs_feat[col].apply(lambda x: '-' if x == query_feat.iloc[0][col] else x)
        # cfs_feat['Type'] = 'Counterfactual'
        # query_feat['Type'] = 'Factual'
        
        # #cfs = cfs.to_parquet(None)
        # cfs_feat = pd.concat([query_feat,cfs_feat])

        return xai_service_pb2.InitializationResponse(


            feature_explanation = xai_service_pb2.Feature_Explanation(
                                    feature_names=train.columns.tolist(),
                                    plots={'pdp': dispatch_table.get(('featureExplanation', 'pdp')).handle(request, models, data, model_name, 'featureExplanation'),
                                            'ale': dispatch_table.get(('featureExplanation', 'ale')).handle(request, models, data, model_name, 'featureExplanation'),     
                                            },
                                # tables = {'counterfactuals': xai_service_pb2.ExplanationsResponse(
                                #             explainability_type = 'featureExplanation',
                                #             explanation_method = 'counterfactuals',
                                #             explainability_model = model_id,
                                #             plot_name = 'Counterfactual Explanations',
                                #             plot_descr = "Counterfactual Explanations identify the minimal changes needed to alter a machine learning model's prediction for a given instance.",
                                #             plot_type = 'Table',
                                #             table_contents = {col: xai_service_pb2.TableContents(index=i+1,values=cfs_feat[col].astype(str).tolist()) for i,col in enumerate(cfs_feat.columns)}
                                #         )}     
                            ),

            hyperparameter_explanation = xai_service_pb2.Hyperparameter_Explanation(
                                    hyperparameter_names=list(original_model.param_grid.keys()),
                                    plots={'pdp': dispatch_table.get(('featureExplanation', 'pdp')).handle(request, models, data, model_name, 'hyperparameterExplanation'),
                                           '2dpdp': dispatch_table.get(('featureExplanation', '2dpdp')).handle(request, models, data, model_name, 'hyperparameterExplanation'),
                                            'ale': dispatch_table.get(('featureExplanation', 'ale')).handle(request, models, data, model_name, 'hyperparameterExplanation'),     
                                            },   
            
                                # tables = {'counterfactuals': xai_service_pb2.ExplanationsResponse(
                                #             explainability_type = 'hyperparameterExplanation',
                                #             explanation_method = 'counterfactuals',
                                #             explainability_model = model_id,
                                #             plot_name = 'Counterfactual Explanations',
                                #             plot_descr = "Counterfactual Explanations identify the minimal changes on hyperparameter values in order to correctly classify a given missclassified instance.",
                                #             plot_type = 'Table',
                                #             table_contents = {col: xai_service_pb2.TableContents(index=i+1,values=cfs[col].astype(str).tolist()) for i,col in enumerate(cfs.columns)}
                                #         )}     
                            ),
                )



    def ModelAnalysisTask(self, request, context):
        models = json.load(open("metadata/models.json"))
        data = json.load(open("metadata/datasets.json"))
        model_name = request.model_name
        model_id = request.model_id

        dispatch_table = {
            ('hyperparameterExplanation', 'pdp'): PDPHandler(),
            ('hyperparameterExplanation', '2dpdp'): TwoDPDPHandler(),
            ('hyperparameterExplanation', 'ale'): ALEHandler(),
            ('hyperparameterExplanation', 'counterfactuals'): CounterfactualsHandler(),
            ('featureExplanation', 'pdp'): PDPHandler(),
            ('featureExplanation', '2dpdp'): TwoDPDPHandler(),
            ('featureExplanation', 'ale'): ALEHandler(),
            ('featureExplanation', 'counterfactuals'): CounterfactualsHandler(),
            ('featureExplanation', 'prototypes'): PrototypesHandler()
        }
       # Load trained model if exists
        try:
            with open(models[model_name]['original_model'], 'rb') as f:
                original_model = joblib.load(f)
        except FileNotFoundError:
            print("Model does not exist. Load existing model.")

        try:
            with open(models[model_name]['all_models'], 'rb') as f:
                trained_models = joblib.load(f)
        except FileNotFoundError:
            print("Model does not exist. Load existing model.")

        model = trained_models[model_id]

        # Load Data
        train = pd.read_csv(data[model_name]['train'],index_col=0) 
        train_labels = pd.read_csv(data[model_name]['train_labels'],index_col=0) 
        test = pd.read_csv(data[model_name]['test'],index_col=0) 
        test_labels = pd.read_csv(data[model_name]['test_labels'],index_col=0) 

        test['label'] = test_labels
        # dataframe = pd.concat([train.reset_index(drop=True), train_labels.reset_index(drop=True)], axis = 1)

        # predictions = original_model.predict(test)
        # test['Predicted'] = predictions
        # test['Label'] = (test['label'] != test['Predicted']).astype(int)


        # param_grid = original_model.param_grid
        # param_grid = transform_grid_plt(param_grid)
         

        # # ---------------------- Run Explainability Methods for Model -----------------------------------------------

        # # PD Plots
        # numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        # features = train.columns.tolist()[0]
        # numeric_features = train.select_dtypes(include=numerics).columns.tolist()
        # categorical_features = train.columns.drop(numeric_features)

        # pdp = partial_dependence(model, train, features = [train.columns.tolist().index(features)],
        #                         feature_names=train.columns.tolist(),categorical_features=categorical_features)

        # pdp_grid = [value.tolist() for value in pdp['grid_values']][0]
        # pdp_vals = [value.tolist() for value in pdp['average']][0]

        # #ALE Plots
        # if train[features].dtype in ['int','float']:
        #     ale_eff_feat = ale(X=train, model=model, feature=[features],plot=False, grid_size=50, include_CI=True, C=0.95)
        # else:
        #     ale_eff_feat = ale(X=train, model=model, feature=[features],plot=False, grid_size=50, predictors=train.columns.tolist(), include_CI=True, C=0.95)

        return xai_service_pb2.ModelAnalysisTaskResponse(

            feature_explanation = xai_service_pb2.Feature_Explanation(
                                    feature_names=train.columns.tolist(),
                                    plots={'pdp': dispatch_table.get(('featureExplanation', 'pdp')).handle(request, models, data, model_name, 'featureExplanation'),
                                            'ale': dispatch_table.get(('featureExplanation', 'ale')).handle(request, models, data, model_name, 'featureExplanation'),     
                                            },
                                ),            
                )
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    xai_service_pb2_grpc.add_ExplanationsServicer_to_server(ExplainabilityExecutor(), server)
    #xai_service_pb2_grpc.add_InfluencesServicer_to_server(MyInfluencesService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
