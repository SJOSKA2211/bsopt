import React, { useState, useEffect } from 'react';
// Import Apollo Client hooks and gql tag
import { useQuery, useMutation, gql } from '@apollo/client';

// Import custom GraphQL hooks
import { useFetchDataGQL, useMutateDataGQL } from '../hooks/api'; 

// --- GraphQL Queries and Mutations ---
// Query to fetch ML models
const GET_ML_MODELS_QUERY = gql`
  query GetMlModels {
    mlModels { # Assuming a 'mlModels' query exists at the backend
      id
      name
      version
      description
      isActive
    }
  }
`;

// Mutation to create an ML model
const CREATE_ML_MODEL_MUTATION = gql`
  mutation CreateMLModel($name: String!, $version: String!, $description: String, $isActive: Boolean) {
    createMlModel(name: $name, version: $version, description: $description, isActive: $isActive) { # Assuming mutation structure
      id
      name
      version
      description
      isActive
    }
  }
`;

// Mutation to trigger ML model prediction
const PREDICT_ML_MODEL_MUTATION = gql`
  mutation PredictMLModel($modelId: String!, $data: JSONObject!) { # Assuming input types
    predict(modelId: $modelId, data: $data) {
      prediction
      confidence
      modelUsed
      timestamp
    }
  }
`;

// Mutation to trigger ML model training
const TRAIN_ML_MODEL_MUTATION = gql`
  mutation TriggerTraining($modelId: String!, $trainingParams: TrainingParamsInput!) { # Assuming input types
    triggerTraining(modelId: $modelId, trainingParams: $trainingParams) {
      message
      modelId
      status
      timestamp
    }
  }
`;

// Mutation to trigger ML model deployment
const DEPLOY_ML_MODEL_MUTATION = gql`
  mutation DeployMLModel($modelId: String!, $deploymentParams: DeploymentParamsInput!) { # Assuming input types
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
  const { data: mlModelsData, loading: modelsLoading, error: modelsError, refetch: refetchModels } = useFetchDataGQL<any>(GET_ML_MODELS_QUERY);

  // Mutations
  const [createModel, { loading: creatingModel, error: createModelError }] = useMutateDataGQL<any>(CREATE_ML_MODEL_MUTATION);
  const [predict, { data: predictionResult, loading: predicting, error: predictionError }] = useMutateDataGQL<any>(PREDICT_ML_MODEL_MUTATION);
  const [triggerTraining, { data: trainingStatus, loading: training, error: trainingError }] = useMutateDataGQL<any>(TRAIN_ML_MODEL_MUTATION);
  const [deployModel] = useMutateDataGQL<any>(DEPLOY_ML_MODEL_MUTATION);

  const models = mlModelsData?.mlModels || [];

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
      await createModel({
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
      alert(`Prediction: ${response.prediction.prediction} (Confidence: ${response.prediction.confidence})`);
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
      alert(`Training task enqueued: ${response.message}`);
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
        alert(`Deployment task enqueued: ${response.message}`);
    } catch (err: any) {
        console.error(`Failed to deploy model ${deployModelId} v${deployVersion}:`, err);
        alert(`Error deploying model: ${err.message}`);
    }
  };

  return (
    <div>
      <h1>ML Pipeline Management</h1>

      {/* Model Creation */}
      <div>
        <h2>Manage ML Models</h2>
        <input type="text" placeholder="Model Name" value={newModelName} onChange={(e) => setNewModelName(e.target.value)} />
        <input type="text" placeholder="Version (e.g., 1.0.0)" value={newModelVersion} onChange={(e) => setNewModelVersion(e.target.value)} />
        <input type="text" placeholder="Description" value={newModelDescription} onChange={(e) => setNewModelDescription(e.target.value)} />
        <button onClick={handleCreateModel} disabled={!newModelName || !newModelVersion || creatingModel}>
          {creatingModel ? 'Creating...' : 'Add ML Model'}
        </button>
        {createModelError && <p style={{ color: 'red' }}>Error: {createModelError}</p>}
      </div>

      {/* Model List and Actions */}
      <div>
        <h2>Active ML Models</h2>
        {modelsLoading && <p>Loading models...</p>}
        {modelsError && <p>Error loading models: {modelsError}</p>}
        {models && models.length > 0 ? (
          <ul>
            {models.map((model) => (
              <li key={model.id}>
                {model.name} v{model.version} - {model.description || 'No description'}
                <button onClick={() => setSelectedModelId(model.id)} disabled={selectedModelId === model.id}>
                  Select for Prediction
                </button>
                <button onClick={() => setTrainingModelId(model.id)}>Select for Training</button>
                <button onClick={() => { setDeployModelId(model.id); setDeployVersion(model.version); }}>Select for Deployment</button>
              </li>
            ))}
          </ul>
        ) : (
          <p>No active ML models found.</p>
        )}
      </div>

      {/* Prediction Section */}
      {selectedModelId && (
        <div>
          <h3>Predict using {models.find(m => m.id === selectedModelId)?.name} v{models.find(m => m.id === selectedModelId)?.version}</h3>
          <input type="text" placeholder="Input Value for Prediction" value={predictionInput} onChange={(e) => setPredictionInput(e.target.value)} />
          <button onClick={handlePredict} disabled={predicting}>
            {predicting ? 'Predicting...' : 'Get Prediction'}
          </button>
          {predictionError && <p style={{ color: 'red' }}>Error: {predictionError}</p>}
        </div>
      )}

      {/* Training Trigger Section */}
      {trainingModelId && (
        <div>
          <h3>Trigger Training</h3>
          <input type="number" placeholder="Epochs" value={epochs} onChange={(e) => setEpochs(parseInt(e.target.value) || 0)} />
          <input type="number" placeholder="Batch Size" value={batchSize} onChange={(e) => setBatchSize(parseInt(e.target.value) || 0)} />
          <button onClick={handleTriggerTraining} disabled={training}>
            {training ? 'Starting Training...' : 'Start Training'}
          </button>
          {trainingError && <p style={{ color: 'red' }}>Error: {trainingError}</p>}
        </div>
      )}

       {/* Deployment Trigger Section */}
      {deployModelId && (
        <div>
          <h3>Deploy Model</h3>
          <p>Model: {models.find(m => m.id === deployModelId)?.name} v{models.find(m => m.id === deployModelId)?.version}</p>
          <select value={deployEnv} onChange={(e) => setDeployEnv(e.target.value)}>
            <option value="staging">Staging</option>
            <option value="production">Production</option>
          </select>
          <button onClick={handleDeployModel} disabled={/* deploy loading state if available */ false}>
            Deploy Model
          </button>
          {/* Add error handling for deployment */}
        </div>
      )}
    </div>
  );
};

export default MLPage;
