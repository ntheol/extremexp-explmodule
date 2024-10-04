import xai_service_pb2_grpc
import xai_service_pb2
import joblib
import pickle
from modules.lib import *
from modules.pdp import partial_dependence_1D,partial_dependence_2D
from modules.ALE_generic import ale 
from sklearn.inspection import partial_dependence
import ast
import dice_ml
from aix360.algorithms.protodash import ProtodashExplainer

class BaseExplanationHandler:
    """Base class for all explanation handlers."""
    
    def handle(self, request, models, data, model_name, explanation_type):
        raise NotImplementedError("Subclasses should implement this method")

    def _load_model(self, model_path, model_name):
        """Helper to load model (same as before)."""
        try:
            with open(model_path, 'rb') as f:
                if model_name == 'Ideko_model':
                    return pickle.load(f)
                else:
                    return joblib.load(f)
        except FileNotFoundError:
            print(f"Model '{model_path}' does not exist.")
            return None

    def _load_or_train_surrogate_model(self, models, model_name, original_model, param_grid):
        """Helper to load or train surrogate model (same as before)."""
        try:
            with open(models[model_name]['pdp_ale_surrogate_model'], 'rb') as f:
                return joblib.load(f)
        except FileNotFoundError:
            print("Surrogate model does not exist. Training a new one.")
            surrogate_model = proxy_model(param_grid, original_model, 'accuracy', 'XGBoostRegressor')
            joblib.dump(surrogate_model, models[model_name]['pdp_ale_surrogate_model'])
            return surrogate_model
        
    def _load_or_train_cf_surrogate_model(self, models, model_name, original_model, train, train_labels,query):
        # try:
        #     with open(models[model_name]['cfs_surrogate_model'], 'rb') as f:
        #         surrogate_model = joblib.load(f)
        #         proxy_dataset = pd.read_csv(models[model_name]['cfs_surrogate_dataset'],index_col=0)
        # except FileNotFoundError:
        print("Surrogate model does not exist. Training new surrogate model")
        surrogate_model , proxy_dataset = instance_proxy(train,train_labels,original_model, query.loc[0],original_model.param_grid)
            # joblib.dump(surrogate_model, models[model_name]['cfs_surrogate_model'])  
            # proxy_dataset.to_csv(models[model_name]['cfs_surrogate_dataset'])
        return surrogate_model, proxy_dataset

class PDPHandler(BaseExplanationHandler):

    def handle(self, request, models, data, model_name, explanation_type):

        if explanation_type == 'featureExplanation':
            model_id = request.model_id
            original_model = self._load_model(models[model_name]['original_model'], model_name)
            trained_models = self._load_model(models[model_name]['all_models'], model_name)
            model = trained_models[model_id]
            dataframe = pd.DataFrame()
            dataframe = pd.read_csv(data[model_name]['train'],index_col=0) 
            features = request.feature1
            numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
            numeric_features = dataframe.select_dtypes(include=numerics).columns.tolist()
            categorical_features = dataframe.columns.drop(numeric_features)

            pdp = partial_dependence(model, dataframe, features = [dataframe.columns.tolist().index(features)],
                                    feature_names=dataframe.columns.tolist(),categorical_features=categorical_features)
            
            if type(pdp['grid_values'][0][0]) == str:
                axis_type='categorical' 
            else: axis_type = 'numerical'

            pdp_grid = [value.tolist() for value in pdp['grid_values']][0]
            pdp_vals = [value.tolist() for value in pdp['average']][0]
            return xai_service_pb2.ExplanationsResponse(
                explainability_type = explanation_type,
                explanation_method = 'pdp',
                explainability_model = model_name,
                plot_name = 'Partial Dependence Plot (PDP)',
                plot_descr = "PD (Partial Dependence) Plots show how a feature affects a model's predictions, holding other features constant, to illustrate feature impact.",
                plot_type = 'LinePlot',
                features = xai_service_pb2.Features(
                            feature1=features, 
                            feature2=''),
                xAxis = xai_service_pb2.Axis(
                            axis_name=f'{features}', 
                            axis_values=[str(value) for value in pdp_grid], 
                            axis_type=axis_type  
                ),
                yAxis = xai_service_pb2.Axis(
                            axis_name='PDP Values', 
                            axis_values=[str(value) for value in pdp_vals], 
                            axis_type='numerical'
                ),
                zAxis = xai_service_pb2.Axis(
                            axis_name='', 
                            axis_values='', 
                            axis_type=''                    
                )
            )
        else:
            feature = request.feature1
            original_model = self._load_model(models[model_name]['original_model'], model_name)
            param_grid = transform_grid_plt(original_model.param_grid)
            surrogate_model = self._load_or_train_surrogate_model(models, model_name, original_model, param_grid)

            param_grid = transform_grid(param_grid)
            param_space, name = dimensions_aslists(param_grid)
            space = Space(param_space)

            feats = {}
            for index,n in enumerate(name):
                feats[n] = index

            plot_dims = []
            for row in range(space.n_dims):
                # if space.dimensions[row].is_constant:
                #     continue
                plot_dims.append((row, space.dimensions[row]))
                
            pdp_samples = space.rvs(n_samples=1000,random_state=123456)


            xi = []
            yi=[]
            index, dim = plot_dims[feats[feature]]
            xi1, yi1 = partial_dependence_1D(space, surrogate_model,
                                                index,
                                                samples=pdp_samples,
                                                n_points=100)

            xi.append(xi1)
            yi.append(yi1)
                
            x = [arr.tolist() for arr in xi]
            y = [arr for arr in yi]
            axis_type = 'categorical' if isinstance(x[0][0], str) else 'numerical'

            return xai_service_pb2.ExplanationsResponse(
                explainability_type=explanation_type,
                explanation_method='pdp',
                explainability_model=model_name,
                plot_name='Partial Dependence Plot (PDP)',
                plot_descr="PD (Partial Dependence) Plots show how different hyperparameter values affect a model's accuracy, holding other hyperparameters constant.",
                plot_type='LinePlot',
                features=xai_service_pb2.Features(
                    feature1=feature, 
                    feature2=''
                ),
                xAxis=xai_service_pb2.Axis(
                    axis_name=f'{feature}',
                    axis_values=[str(value) for value in x[0]],
                    axis_type=axis_type
                ),
                yAxis=xai_service_pb2.Axis(
                    axis_name='PDP Values',
                    axis_values=[str(value) for value in y[0]],
                    axis_type='numerical'
                ),
                zAxis=xai_service_pb2.Axis(
                    axis_name='',
                    axis_values='',
                    axis_type=''
                )
            )

class TwoDPDPHandler(BaseExplanationHandler):

    def handle(self, request, models, data, model_name, explanation_type):
        if explanation_type == 'featureExplanation':
            feature1 = request.feature1
            feature2 = request.feature2
            model_id = request.model_id
            original_model = self._load_model(models[model_name]['original_model'], model_name)
            trained_models = self._load_model(models[model_name]['all_models'], model_name)
            model = trained_models[model_id]
            dataframe = pd.read_csv(data[model_name]['train'],index_col=0)                        
            feature1 = request.feature1
            feature2 = request.feature2
            numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

            numeric_features = dataframe.select_dtypes(include=numerics).columns.tolist()
            categorical_features = dataframe.columns.drop(numeric_features)

            pdp = partial_dependence(model, dataframe, features = [(dataframe.columns.tolist().index(feature1),dataframe.columns.tolist().index(feature2))],
                                    feature_names=dataframe.columns.tolist(),categorical_features=categorical_features)
            

            if type(pdp['grid_values'][0][0]) == str:
                axis_type_0='categorical' 
            else: axis_type_0 = 'numerical'

            if type(pdp['grid_values'][1][0]) == str:
                axis_type_1='categorical' 
            else: axis_type_1 = 'numerical'


            pdp_grid_1 = [value.tolist() for value in pdp['grid_values']][0]
            pdp_grid_2 = [value.tolist() for value in pdp['grid_values']][1]
            pdp_vals = [value.tolist() for value in pdp['average']][0]
            return xai_service_pb2.ExplanationsResponse(
                explainability_type = explanation_type,
                explanation_method = '2dpdp',
                explainability_model = model_name,
                plot_name = '2D-Partial Dependence Plot (2D-PDP)',
                plot_descr = "2D-PD plots visualize how the model's accuracy changes when two hyperparameters vary.",
                plot_type = 'ContourPlot',
                features = xai_service_pb2.Features(
                            feature1=feature1, 
                            feature2=feature2),
                xAxis = xai_service_pb2.Axis(
                            axis_name=f'{feature1}', 
                            axis_values=[str(value) for value in pdp_grid_1], 
                            axis_type=axis_type_0  
                ),
                yAxis = xai_service_pb2.Axis(
                            axis_name=f'{feature2}', 
                            axis_values=[str(value) for value in pdp_grid_2], 
                            axis_type=axis_type_1
                ),
                zAxis = xai_service_pb2.Axis(
                            axis_name='', 
                            axis_values=[str(value) for value in pdp_vals], 
                            axis_type='numerical'                    
                )
            )
        else:
            feature1 = request.feature1
            feature2 = request.feature2
            original_model = self._load_model(models[model_name]['original_model'], model_name)
            param_grid = transform_grid_plt(original_model.param_grid)
            surrogate_model = self._load_or_train_surrogate_model(models, model_name, original_model, param_grid)

            param_space, name = dimensions_aslists(param_grid)
            space = Space(param_space)

            index1 = name.index(feature1)
            index2 = name.index(feature2)


            plot_dims = []
            for row in range(space.n_dims):
                if space.dimensions[row].is_constant:
                    continue
                plot_dims.append((row, space.dimensions[row]))
            
            pdp_samples = space.rvs(n_samples=1000,random_state=123456)

            _ ,dim_1 = plot_dims[index1]
            _ ,dim_2 = plot_dims[index2]
            xi, yi, zi = partial_dependence_2D(space, surrogate_model,
                                                    index1, index2,
                                                    pdp_samples, 100)
            
            
            x = [arr.tolist() for arr in xi]
            y = [arr.tolist() for arr in yi]
            z = [arr.tolist() for arr in zi]

            return xai_service_pb2.ExplanationsResponse(
                        explainability_type = explanation_type,
                        explanation_method = '2dpdp',
                        explainability_model = model_name,
                        plot_name = '2D-Partial Dependence Plot (2D-PDP)',
                        plot_descr = "2D-PD plots visualize how the model's accuracy changes when two hyperparameters vary.",
                        plot_type = 'ContourPlot',
                        features = xai_service_pb2.Features(
                                    feature1=feature1, 
                                    feature2=feature2),
                        xAxis = xai_service_pb2.Axis(
                                    axis_name=f'{feature2}', 
                                    axis_values=[str(value) for value in x], 
                                    axis_type='categorical' if isinstance(x[0], str) else 'numerical'
                        ),
                        yAxis = xai_service_pb2.Axis(
                                    axis_name=f'{feature1}', 
                                    axis_values=[str(value) for value in y], 
                                    axis_type='categorical' if isinstance(y[0], str) else 'numerical'
                        ),
                        zAxis = xai_service_pb2.Axis(
                                    axis_name='', 
                                    axis_values=[str(value) for value in z], 
                                    axis_type='numerical' 
                        )
            )
    
class ALEHandler(BaseExplanationHandler):

    def handle(self, request, models, data, model_name, explanation_type):
        if explanation_type == 'featureExplanation':
            features = request.feature1
            original_model = self._load_model(models[model_name]['original_model'], model_name)
            trained_models = self._load_model(models[model_name]['all_models'], model_name)
            model_id = request.model_id
            model = trained_models[model_id]

            dataframe = pd.read_csv(data[model_name]['train'],index_col=0) 
            features = request.feature1

            if dataframe[features].dtype in ['int','float']:
                ale_eff = ale(X=dataframe, model=model, feature=[features],plot=False, grid_size=50, include_CI=True, C=0.95)
            else:
                ale_eff = ale(X=dataframe, model=model, feature=[features],plot=False, grid_size=50, predictors=dataframe.columns.tolist(), include_CI=True, C=0.95)

            return xai_service_pb2.ExplanationsResponse(
                explainability_type = explanation_type,
                explanation_method = 'ale',
                explainability_model = model_name,
                plot_name = 'Accumulated Local Effects Plot (ALE)',
                plot_descr = "ALE plots illustrate the effect of a single feature on the predicted outcome of a machine learning model.",
                plot_type = 'LinePLot',
                features = xai_service_pb2.Features(
                            feature1=features, 
                            feature2=''),
                xAxis = xai_service_pb2.Axis(
                            axis_name=f'{features}', 
                            axis_values=[str(value) for value in ale_eff.index.tolist()], 
                            axis_type='categorical' if isinstance(ale_eff.index.tolist()[0], str) else 'numerical'
                ),
                yAxis = xai_service_pb2.Axis(
                            axis_name='ALE Values', 
                            axis_values=[str(value) for value in ale_eff.eff.tolist()], 
                            axis_type='categorical' if isinstance(ale_eff.eff.tolist()[0], str) else 'numerical'
                ),
                zAxis = xai_service_pb2.Axis(
                            axis_name='', 
                            axis_values='', 
                            axis_type=''                    
                )
            )
        else:
            feature1 = request.feature1
            original_model = self._load_model(models[model_name]['original_model'], model_name)
            param_grid = transform_grid(original_model.param_grid)
            surrogate_model = self._load_or_train_surrogate_model(models, model_name, original_model, param_grid)

            param_space, name = dimensions_aslists(param_grid)
            space = Space(param_space)

            plot_dims = []
            for row in range(space.n_dims):
                if space.dimensions[row].is_constant:
                    continue
                plot_dims.append((row, space.dimensions[row]))

            pdp_samples = space.rvs(n_samples=1000,random_state=123456)
            data = pd.DataFrame(pdp_samples,columns=[n for n in name])

            if data[feature1].dtype in ['int','float']:
                # data = data.drop(columns=feat)
                # data[feat] = d1[feat]  
                ale_eff = ale(X=data, model=surrogate_model, feature=[feature1],plot=False, grid_size=50, include_CI=True, C=0.95)
            else:
                ale_eff = ale(X=data, model=surrogate_model, feature=[feature1],plot=False, grid_size=50,predictors=data.columns.tolist(), include_CI=True, C=0.95)

            return xai_service_pb2.ExplanationsResponse(
                explainability_type = explanation_type,
                explanation_method = 'ale',
                explainability_model = model_name,
                plot_name = 'Accumulated Local Effects Plot (ALE)',
                plot_descr = "ALE Plots illustrate the effect of a single hyperparameter on the accuracy of a machine learning model.",
                plot_type = 'LinePLot',
                features = xai_service_pb2.Features(
                            feature1=feature1, 
                            feature2=''),
                xAxis = xai_service_pb2.Axis(
                            axis_name=f'{feature1}', 
                            axis_values=[str(value) for value in ale_eff.index.tolist()], 
                            axis_type='categorical' if isinstance(ale_eff.index.tolist()[0], str) else 'numerical'
                ),
                yAxis = xai_service_pb2.Axis(
                            axis_name='ALE Values', 
                            axis_values=[str(value) for value in ale_eff.eff.tolist()], 
                            axis_type='categorical' if isinstance(ale_eff.eff.tolist()[0], str) else 'numerical'
                ),
                zAxis = xai_service_pb2.Axis(
                            axis_name='', 
                            axis_values='', 
                            axis_type=''                    
                )
            )
        
class CounterfactualsHandler(BaseExplanationHandler):

    def handle(self, request, models, data, model_name, explanation_type):
        
        if explanation_type == 'featureExplanation':
            model_id = request.model_id
            query = request.query
            query = ast.literal_eval(query)
            query = pd.DataFrame([query])

            query = query.drop(columns=['id','label'])
            original_model = self._load_model(models[model_name]['original_model'], model_name)
            trained_models = self._load_model(models[model_name]['all_models'], model_name)
            model = trained_models[model_id]

            target = request.target


            train = pd.read_csv(data[model_name]['train'],index_col=0)  
            train_labels = pd.read_csv(data[model_name]['train_labels'],index_col=0)  
            
            dataframe = pd.concat([train.reset_index(drop=True), train_labels.reset_index(drop=True)], axis = 1)

            d = dice_ml.Data(dataframe=dataframe, 
                continuous_features=dataframe.drop(columns=target).select_dtypes(include='number').columns.tolist()
                , outcome_name=target)
    
            # Using sklearn backend
            m = dice_ml.Model(model=original_model, backend="sklearn")
            # Using method=random for generating CFs
            exp = dice_ml.Dice(d, m, method="random")
            e1 = exp.generate_counterfactuals(query.drop(columns=['prediction']), total_CFs=5, desired_class="opposite",sample_size=5000)
            e1.visualize_as_dataframe(show_only_changes=True)
            cfs = e1.cf_examples_list[0].final_cfs_df
            query.rename(columns={"prediction": target},inplace=True)
            # for col in query.columns:
            #     cfs[col] = cfs[col].apply(lambda x: '-' if x == query.iloc[0][col] else x)
            cfs['Type'] = 'Counterfactual'
            query['Type'] = 'Factual'
            
            #cfs = cfs.to_parquet(None)
            cfs = pd.concat([query,cfs])

            return xai_service_pb2.ExplanationsResponse(
                explainability_type = explanation_type,
                explanation_method = 'counterfactuals',
                explainability_model = model_name,
                plot_name = 'Counterfactual Explanations',
                plot_descr = "Counterfactual Explanations identify the minimal changes needed to alter a machine learning model's prediction for a given instance.",
                plot_type = 'Table',
                table_contents = {col: xai_service_pb2.TableContents(index=i+1,values=cfs[col].astype(str).tolist()) for i,col in enumerate(cfs.columns)}
            )
        
        else:
            original_model = self._load_model(models[model_name]['original_model'], model_name)
            model_id = request.model_id
            query = request.query
            
            query = ast.literal_eval(query)
            query = pd.DataFrame([query])
            trained_models = self._load_model(models[model_name]['all_models'], model_name)
            model = trained_models[model_id]
            train = pd.read_csv(data[model_name]['train'],index_col=0) 
            train_labels = pd.read_csv(data[model_name]['train_labels'],index_col=0) 
            surrogate_model , proxy_dataset = self._load_or_train_cf_surrogate_model(models, model_name, original_model, train, train_labels,query)
            param_grid = transform_grid(original_model.param_grid)
            param_space, name = dimensions_aslists(param_grid)
            space = Space(param_space)

            plot_dims = []
            for row in range(space.n_dims):
                if space.dimensions[row].is_constant:
                    continue
                plot_dims.append((row, space.dimensions[row]))
            iscat = [isinstance(dim[1], Categorical) for dim in plot_dims]
            categorical = [name[i] for i,value in enumerate(iscat) if value == True]
            proxy_dataset[categorical] = proxy_dataset[categorical].astype(str)


            params = model.get_params()
            query = pd.DataFrame(data = {'Model__learning_rate':params['Model__learning_rate'], 'Model__max_depth':params['Model__max_depth'],	'Model__min_child_weight':params['Model__min_child_weight'],'Model__n_estimators':params['Model__n_estimators'],	'preprocessor__num__scaler':params['preprocessor__num__scaler']},index=[0])
            query = pd.DataFrame.from_dict(original_model.best_params_,orient='index').T
            query[categorical] = query[categorical].astype(str)
            d = dice_ml.Data(dataframe=proxy_dataset, continuous_features=proxy_dataset.drop(columns='BinaryLabel').select_dtypes(include='number').columns.tolist(), outcome_name='BinaryLabel')
            # Using sklearn backend
            m = dice_ml.Model(model=surrogate_model, backend="sklearn")
            # Using method=random for generating CFs
            exp = dice_ml.Dice(d, m, method="random")
            e1 = exp.generate_counterfactuals(query, total_CFs=5, desired_class="opposite",sample_size=5000)
            dtypes_dict = proxy_dataset.drop(columns='BinaryLabel').dtypes.to_dict()
            cfs = e1.cf_examples_list[0].final_cfs_df
            for col, dtype in dtypes_dict.items():
                cfs[col] = cfs[col].astype(dtype)
            scaled_query, scaled_cfs = min_max_scale(proxy_dataset=proxy_dataset,factual=query.copy(deep=True),counterfactuals=cfs.copy(deep=True),label='BinaryLabel')
            cfs['Cost'] = cf_difference(scaled_query, scaled_cfs)
            cfs = cfs.sort_values(by='Cost')
            cfs['Type'] = 'Counterfactual'
            #query['BinaryLabel'] = 1
            query['Cost'] = '-'
            query['Type'] = 'Factual'
            query['BinaryLabel'] = 1
            for col in query.columns:
                cfs[col] = cfs[col].apply(lambda x: '-' if x == query.iloc[0][col] else x)
                cfs = pd.concat([query,cfs])

            return xai_service_pb2.ExplanationsResponse(
                explainability_type = explanation_type,
                explanation_method = 'counterfactuals',
                explainability_model = model_name,
                plot_name = 'Counterfactual Explanations',
                plot_descr = "Counterfactual Explanations identify the minimal changes on hyperparameter values in order to correctly classify a given missclassified instance.",
                plot_type = 'Table',
                table_contents = {col: xai_service_pb2.TableContents(index=i+1,values=cfs[col].astype(str).tolist()) for i,col in enumerate(cfs.columns)}
            )
        

class PrototypesHandler(BaseExplanationHandler):

    def handle(self, request, models, data, model_name, explanation_type):
        model_id = request.model_id
        query = request.query
        query = ast.literal_eval(query)
        query = pd.DataFrame([query])
        trained_models = self._load_model(models[model_name]['all_models'], model_name)
        model = trained_models[model_id]
        train = pd.read_csv(data[model_name]['train'],index_col=0) 
        train_labels = pd.read_csv(data[model_name]['train_labels'],index_col=0) 
        train['label'] = train_labels
        
        explainer = ProtodashExplainer()
        reference_set_train = train[train.label==0].drop(columns='label')

        (W, S, _)= explainer.explain(np.array(query.drop(columns='predictions')).reshape(1,-1),np.array(reference_set_train),m=5)
        prototypes = reference_set_train.reset_index(drop=True).iloc[S, :].copy()
        prototypes['predictions'] =  model.predict(prototypes)
        prototypes = prototypes.reset_index(drop=True).T
        prototypes.rename(columns={0:'Prototype1',1:'Prototype2',2:'Prototype3',3:'Prototype4',4:'Prototype5'},inplace=True)
        prototypes = prototypes.reset_index()

        prototypes.set_index('index', inplace=True)

        # Create a new empty dataframe for boolean results
        boolean_df = pd.DataFrame(index=prototypes.index)

        # Iterate over each column and compare with the series
        for col in prototypes.columns:
            boolean_df[col] = prototypes[col] == query.loc[0][prototypes.index].values

        prototypes.reset_index(inplace=True)
        prototypes= prototypes.append([{'index': 'Weights', 'Prototype1':np.around(W/np.sum(W), 2)[0],'Prototype2':np.around(W/np.sum(W), 2)[1],'Prototype3':np.around(W/np.sum(W), 2)[2],'Prototype4':np.around(W/np.sum(W), 2)[3],'Prototype5':np.around(W/np.sum(W), 2)[4]}])
        boolean_df=boolean_df.append([{'index': 'Weights', 'Prototype1':False,'Prototype2':False,'Prototype3':False,'Prototype4':False,'Prototype5':False}])

        print(prototypes)
        # Create table_contents dictionary for prototypes
        table_contents =  {col: xai_service_pb2.TableContents(index=i+1,values=prototypes[col].astype(str).tolist(),colour =boolean_df[col].astype(str).tolist()) for i,col in enumerate(prototypes.columns)}


        return xai_service_pb2.ExplanationsResponse(
            explainability_type = explanation_type,
            explanation_method = 'prototypes',
            explainability_model = model_name,
            plot_name = 'Prototypes',
            plot_descr = "Prototypes are prototypical examples that capture the underlying distribution of a dataset. It also weights each prototype to quantify how well it represents the data.",
            plot_type = 'Table',
            table_contents = table_contents
        )