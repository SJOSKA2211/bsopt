import React from 'react';
import {
  Box,
  Typography,
  Stack,
  Chip,
  CircularProgress,
  Divider,
  alpha,
  useTheme,
} from '@mui/material';
import {
  Psychology,
  AutoGraph,
  Update,
} from '@mui/icons-material';
import { useQuery } from '@apollo/client/react';
import { gql } from '@apollo/client';

interface MLPredictionsProps {
  symbol: string;
}



export const GET_ML_PREDICTION = gql`
  query GetMLPrediction($symbol: String!) {
    mlPrediction(symbol: $symbol) {
      symbol
      predictedPrice: predicted_price
      confidenceInterval: confidence_interval
      drift
      modelName: model_name
      lastUpdated: last_updated
    }
  }
`;

export const MLPredictions: React.FC<MLPredictionsProps> = React.memo(({ symbol }) => {

  const theme = useTheme();

  const { data: gqlData, loading: isLoading, error } = useQuery(GET_ML_PREDICTION, {
    variables: { symbol },
    pollInterval: 10000,
  });

  interface GqlData {
    mlPrediction?: {
      predictedPrice: number;
      confidenceInterval: [number, number];
      drift: number;
      modelName: string;
      lastUpdated: string;
    };
  }

  const data = (gqlData as GqlData)?.mlPrediction;



  if (isLoading) {



    return (



      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', minHeight: 200 }}>



        <CircularProgress size={30} aria-label="Loading predictions" />



      </Box>



    );



  }



  if (error || !data) {

    return (

      <Box sx={{ p: 2, textAlign: 'center' }}>

        <Typography color="error" variant="body2">ML Engine unavailable</Typography>

      </Box>

    );

  }



  const { predictedPrice, confidenceInterval, drift, modelName, lastUpdated } = data;

  const isPositive = drift >= 0;



  return (

    <Box sx={{ p: 2 }}>

      <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 2 }}>

        <Psychology sx={{ color: 'secondary.main' }} />

        <Typography variant="subtitle1" fontWeight="bold">

          ML Price Prediction

        </Typography>

      </Stack>



      <Stack spacing={2}>

        <Box>

          <Typography variant="caption" color="text.secondary" sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>

            <AutoGraph sx={{ fontSize: 14 }} /> Target Price (24h)

          </Typography>

          <Box>

            <Typography variant="h4" fontWeight="bold" color="primary.main">

              ${predictedPrice.toLocaleString(undefined, { minimumFractionDigits: 2 })}

            </Typography>

          </Box>

        </Box>



        <Stack direction="row" spacing={1} alignItems="center">

          <Chip

            label={`${isPositive ? '+' : ''}${(drift * 100).toFixed(2)}% Predicted Drift`}

            size="small"

            color={isPositive ? 'success' : 'error'}

            variant="outlined"

            sx={{ fontWeight: 'bold' }}

          />

        </Stack>



        <Box sx={{ bgcolor: alpha(theme.palette.background.paper, 0.5), p: 1.5, borderRadius: 1, border: `1px solid ${alpha(theme.palette.divider, 0.1)}` }}>

          <Typography variant="caption" color="text.secondary" gutterBottom>

            95% Confidence Interval

          </Typography>

          <Typography variant="body2" fontWeight="medium">

            ${confidenceInterval[0].toFixed(2)} — ${confidenceInterval[1].toFixed(2)}

          </Typography>

        </Box>



        <Divider sx={{ opacity: 0.1 }} />



        <Stack direction="row" justifyContent="space-between" alignItems="center">

          <Box>

            <Typography variant="caption" color="text.secondary" display="block">

              Model

            </Typography>

            <Typography variant="caption" fontWeight="bold">

              {modelName}

            </Typography>

          </Box>

          <Box sx={{ textAlign: 'right' }}>

            <Typography variant="caption" color="text.secondary" sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>

              <Update sx={{ fontSize: 12 }} /> Updated

            </Typography>

            <Typography variant="caption">

              {new Date(lastUpdated).toLocaleTimeString()}

            </Typography>

          </Box>

        </Stack>

      </Stack>

    </Box>

  );

});