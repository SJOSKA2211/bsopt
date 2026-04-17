import React, { useState, useEffect } from 'react';
import { useQuery, useMutation, gql } from '@apollo/client'; // Import GraphQL hooks
import { ApolloError } from '@apollo/client'; // Import ApolloError for specific error handling

// Import custom GraphQL hooks
import { useFetchDataGQL, useMutateDataGQL } from '../hooks/api'; // Import the GraphQL hooks

// --- GraphQL Queries and Mutations ---
const GET_ML_MODELS_QUERY = gql`
  query GetMlModels {
    mlModels { # Assuming a 'mlModels' query exists
      id
      name
      version
      description
      isActive
    }
  }
`;

const CREATE_ML_MODEL_MUTATION = gql`
  mutation CreateMLModel($name: String!, $version: String!, $description: String, $isActive: Boolean) {
    createMlModel(name: $name, version: $version, description: $description, isActive: $isActive) { 
      id
      name
      version
      description
      isActive
    }
  }
`;

const PREDICT_ML_MODEL_MUTATION = gql`
  mutation PredictMLModel($modelId: String!, $data: JSONObject!) { 
    predict(modelId: $modelId, data: $data) {
      prediction
      confidence
      modelUsed
      timestamp
    }
  }
`;

const TRAIN_ML_MODEL_MUTATION = gql`
  mutation TriggerTraining($modelId: String!, $trainingParams: TrainingParamsInput!) { 
    triggerTraining(modelId: $modelId, trainingParams: $trainingParams) {
      message
      modelId
      status
      timestamp
    }
  }
`;

const DEPLOY_ML_MODEL_MUTATION = gql`
  mutation DeployMLModel($modelId: String!, $deploymentParams: DeploymentParamsInput!) { 
    deployModel(modelId: $modelId, deploymentParams: $deploymentParams) {
      message
      modelId
      version
      targetEnvironment
      status
      timestamp
    }
  }
`;

// --- Page Component ---
const MLPage = () => {
  // State for model creation/management
  const [newModelName, setNewModelName] = useState('');
  const [newModelVersion, setNewModelVersion] = useState('');
  const [newModelDescription, setNewModelDescription] = useState('');

  // State for prediction
  const [selectedModelId, setSelectedModelId] = useState<string>('');
  const [predictionInput, setPredictionInput] = useState<string>('');

  // State for training trigger
  const [trainingModelId, setTrainingModelId] = useState<string>('');
  const [epochs, setEpochs] = useState<number>(10);
  const [batchSize, setBatchSize] = useState<number>(32);

  // State for deployment
  const [deployModelId, setDeployModelId] = useState<string>('');
  const [deployVersion, setDeployVersion] = useState<string>('');
  const [deployEnv, setDeployEnv] = useState<string>('staging');

  // Fetch active ML models using GraphQL hook
  // Corrected query name and adapted data access
  const { data: mlModelsData, loading: modelsLoading, error: modelsError, refetch: refetchModels } = useFetchDataGQL<any>(GET_ML_MODELS_QUERY);

  // Mutations
  const [createModel, { loading: creatingModel, error: createModelError }] = useMutateDataGQL<any>(CREATE_ML_MODEL_MUTATION);
  const [predict, { data: predictionResult, loading: predicting, error: predictionError }] = useMutateDataGQL<any>(PREDICT_ML_MODEL_MUTATION);
  const [triggerTraining, { loading: training, error: trainingError }] = useMutateDataGQL<any>(TRAIN_ML_MODEL_MUTATION);
  const [deployModel, { loading: deploying, error: deployError }] = useMutateDataGQL<any>(DEPLOY_ML_MODEL_MUTATION);

  const models = mlModelsData?.mlModels || []; // Accessing data based on query structure

  useEffect(() => {
    if (models && models.length > 0 && !selectedModelId) {
      setSelectedModelId(models[0].id); 
      setTrainingModelId(models[0].id); 
      setDeployModelId(models[0].id);   
    }
  }, [models, selectedModelId, trainingModelId, deployModelId]); 

  const handleCreateModel = async () => {
    if (!newModelName || !newModelVersion) {
      alert('Please enter model name and version.');
      return;
    }
    try {
      const response = await createModel({
        variables: { name: newModelName, version: newModelVersion, description: newModelDescription, isActive: true },
      });
      alert(`Model "${newModelName} v${newModelVersion}" created successfully!`);
      setNewModelName('');
      setNewModelVersion('');
      setNewModelDescription('');
      refetchModels(); // Refresh the model list
    } catch (err: any) {
      console.error("Failed to create ML model:", err);
      alert(`Error creating ML model: ${err.message}`);
    }
  };

  const handlePredict = async () => {
    if (!selectedModelId || !predictionInput) {
      alert('Please select a model and enter input data.');
      return;
    }
    try {
      const response = await predict({
        variables: { modelId: selectedModelId, data: { inputValue: parseFloat(predictionInput) } }, 
      });
      // Assuming the prediction result structure matches the schema
      alert(`Prediction: ${response.data.predict.prediction} (Confidence: ${response.data.predict.confidence})`);
    } catch (err: any) {
      console.error("Failed to predict:", err);
      alert(`Error predicting: ${err.message}`);
    }
  };

  const handleTriggerTraining = async () => {
    if (!trainingModelId) {
      alert('Please select a model to train.');
      return;
    }
    try {
      const response = await triggerTraining({
        variables: { modelId: trainingModelId, trainingParams: { epochs, batchSize } },
      });
      alert(`Training task enqueued: ${response.data.triggerTraining.message}`);
    } catch (err: any) {
      console.error("Failed to trigger training:", err);
      alert(`Error triggering training: ${err.message}`);
    }
  };

  const handleDeployModel = async () => {
    if (!deployModelId || !deployVersion) {
        alert('Please select a model and version to deploy.');
        return;
    }
    try {
        const response = await deployModel({
            variables: { modelId: deployModelId, deploymentParams: { version: deployVersion, targetEnvironment: deployEnv } },
        });
        alert(`Deployment task enqueued: ${response.data.deployModel.message}`);
    } catch (err: any) {
        console.error(`Failed to deploy model ${deployModelId} v${deployVersion}:`, err);
        alert(`Error deploying model: ${err.message}`);
    }
  };

  return (
    <div className="container mx-auto p-6 bg-bento-bg text-white min-h-screen">
      <h1 className="text-3xl font-bold mb-6">ML Pipeline Management</h1>

      {/* Model Creation */}
      <div className="mb-8 p-6 bg-gray-800 bg-opacity-75 backdrop-blur-md border border-gray-700 rounded-xl shadow-lg">
        <h2 className="text-2xl font-semibold mb-4">Manage ML Models</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          <input
            type="text"
            placeholder="Model Name"
            value={newModelName}
            onChange={(e) => setNewModelName(e.target.value)}
            className="p-3 rounded-lg bg-gray-700 border border-gray-600 focus:outline-none focus:ring-2 focus:ring-mint"
          />
          <input
            type="text"
            placeholder="Version (e.g., 1.0.0)"
            value={newModelVersion}
            onChange={(e) => setNewModelVersion(e.target.value)}
            className="p-3 rounded-lg bg-gray-700 border border-gray-600 focus:outline-none focus:ring-2 focus:ring-mint"
          />
          <input
            type="text"
            placeholder="Description"
            value={newModelDescription}
            onChange={(e) => setNewModelDescription(e.target.value)}
            className="p-3 rounded-lg bg-gray-700 border border-gray-600 focus:outline-none focus:ring-2 focus:ring-mint"
          />
        </div>
        <button 
          onClick={handleCreateModel} 
          disabled={!newModelName || !newModelVersion || creatingModel}
          className="px-6 py-3 bg-mint text-bento-bg font-semibold rounded-lg shadow-md hover:bg-opacity-90 disabled:opacity-50 transition-colors duration-200"
        >
          {creatingModel ? 'Creating...' : 'Add ML Model'}
        </button>
        {createModelError && <p className="text-red-500 mt-4">Error: {createModelError.message}</p>}
      </div>

      {/* Model List and Actions */}
      <div className="mb-8 p-6 bg-gray-800 bg-opacity-75 backdrop-blur-md border border-gray-700 rounded-xl shadow-lg">
        <h2 className="text-2xl font-semibold mb-4">Active ML Models</h2>
        {modelsLoading && <p>Loading models...</p>}
        {modelsError && <p className="text-red-500">Error loading models: {modelsError}</p>}
        {models && models.length > 0 ? (
          <ul className="space-y-4">
            {models.map((model) => (
              <li key={model.id} className="p-4 bg-gray-700 rounded-lg shadow-sm flex flex-col md:flex-row justify-between items-center space-y-2 md:space-y-0 md:space-x-4">
                <div>
                  <p className="text-lg font-semibold">{model.name} v{model.version}</p>
                  <p className="text-sm text-gray-400">{model.description || 'No description'}</p>
                </div>
                <div className="flex flex-wrap gap-2">
                  <button 
                    onClick={() => setSelectedModelId(model.id)} 
                    disabled={selectedModelId === model.id}
                    className="px-4 py-2 bg-blue-600 text-white font-semibold rounded-lg shadow-md hover:bg-blue-700 disabled:opacity-50"
                  >
                    {selectedModelId === model.id ? 'Selected' : 'Select for Prediction'}
                  </button>
                  <button 
                    onClick={() => setTrainingModelId(model.id)} 
                    className="px-4 py-2 bg-green-600 text-white font-semibold rounded-lg shadow-md hover:bg-green-700 disabled:opacity-50"
                  >
                    Select for Training
                  </button>
                  <button 
                    onClick={() => { setDeployModelId(model.id); setDeployVersion(model.version); }} 
                    className="px-4 py-2 bg-purple-600 text-white font-semibold rounded-lg shadow-md hover:bg-purple-700 disabled:opacity-50"
                  >
                    Select for Deployment
                  </button>
                </div>
              </li>
            ))}
          </ul>
        ) : (!modelsLoading && !modelsError && <p>No active ML models found.</p>)}
      </div>

      {/* Prediction Section */}
      {selectedModelId && (
        <div className="mt-8 p-6 bg-gray-800 bg-opacity-75 backdrop-blur-md border border-gray-700 rounded-xl shadow-lg">
          <h3 className="text-2xl font-semibold mb-4">Predict using {models.find(m => m.id === selectedModelId)?.name} v{models.find(m => m.id === selectedModelId)?.version}</h3>
          <input
            type="text"
            placeholder="Input Value for Prediction"
            value={predictionInput}
            onChange={(e) => setPredictionInput(e.target.value)}
            className="p-3 rounded-lg bg-gray-700 border border-gray-600 focus:outline-none focus:ring-2 focus:ring-mint mb-4 w-full"
          />
          <button 
            onClick={handlePredict} 
            disabled={predicting}
            className="px-6 py-3 bg-mint text-bento-bg font-semibold rounded-lg shadow-md hover:bg-opacity-90 disabled:opacity-50"
          >
            {predicting ? 'Predicting...' : 'Get Prediction'}
          </button>
          {predictionError && <p className="text-red-500 mt-4">Error: {predictionError}</p>}
        </div>
      )}

      {/* Training Trigger Section */}
      {trainingModelId && (
        <div className="mt-8 p-6 bg-gray-800 bg-opacity-75 backdrop-blur-md border border-gray-700 rounded-xl shadow-lg">
          <h3 className="text-2xl font-semibold mb-4">Trigger Training</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            <input
              type="number"
              placeholder="Epochs"
              value={epochs}
              onChange={(e) => setEpochs(parseInt(e.target.value) || 0)}
              className="p-3 rounded-lg bg-gray-700 border border-gray-600 focus:outline-none focus:ring-2 focus:ring-mint"
            />
            <input
              type="number"
              placeholder="Batch Size"
              value={batchSize}
              onChange={(e) => setBatchSize(parseInt(e.target.value) || 0)}
              className="p-3 rounded-lg bg-gray-700 border border-gray-600 focus:outline-none focus:ring-2 focus:ring-mint"
            />
          </div>
          <button 
            onClick={handleTriggerTraining} 
            disabled={training}
            className="px-6 py-3 bg-green-600 text-white font-semibold rounded-lg shadow-md hover:bg-green-700 disabled:opacity-50"
          >
            {training ? 'Starting Training...' : 'Start Training'}
          </button>
          {trainingError && <p className="text-red-500 mt-4">Error: {trainingError}</p>}
        </div>
      )}

       {/* Deployment Trigger Section */}
      {deployModelId && (
        <div className="mt-8 p-6 bg-gray-800 bg-opacity-75 backdrop-blur-md border border-gray-700 rounded-xl shadow-lg">
          <h3 className="text-2xl font-semibold mb-4">Deploy Model</h3>
          <p>Model: {models.find(m => m.id === deployModelId)?.name} v{models.find(m => m.id === deployModelId)?.version}</p>
          <select value={deployEnv} onChange={(e) => setDeployEnv(e.target.value)} className="p-3 rounded-lg bg-gray-700 border border-gray-600 focus:outline-none focus:ring-2 focus:ring-mint mb-4">
            <option value="staging">Staging</option>
            <option value="production">Production</option>
          </select>
          <button 
            onClick={handleDeployModel} 
            disabled={deploying}
            className="px-6 py-3 bg-purple-600 text-white font-semibold rounded-lg shadow-md hover:bg-purple-700 disabled:opacity-50 ml-4"
          >
            {deploying ? 'Deploying...' : 'Deploy Model'}
          </button>
          {deployError && <p className="text-red-500 mt-4">Error: {deployError.message}</p>}
        </div>
      )}
    </div>
  );
};

export default MLPage;
