import React, { useState } from 'react';
import { Container, Typography, Paper, Grid, TextField, Button, CircularProgress } from '@mui/material';
import { getOptionPrice } from '../services/pricingService';
import { PriceRequest } from '../types/pricing';

const Analysis: React.FC = () => {
    const [request, setRequest] = useState<PriceRequest>({
        spot: 100,
        strike: 105,
        time_to_expiry: 0.5,
        rate: 0.05,
        volatility: 0.2,
        option_type: 'call',
        model: 'black_scholes',
    });
    const [price, setPrice] = useState<number | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleFetchPrice = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await getOptionPrice(request);
            setPrice(response.price);
        } catch (err: any) {
            setError(err.message || "An error occurred");
        } finally {
            setLoading(false);
        }
    };

    const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const { name, value } = event.target;
        setRequest(prev => ({ ...prev, [name]: parseFloat(value) || value }));
    };

    return (
        <Container maxWidth="lg">
            <Typography variant="h4" gutterBottom>Option Price Analysis</Typography>
            <Paper sx={{ p: 2 }}>
                <Grid container spacing={2}>
                    <Grid item xs={12} sm={6}>
                        <TextField name="spot" label="Spot Price" type="number" value={request.spot} onChange={handleChange} fullWidth />
                    </Grid>
                    <Grid item xs={12} sm={6}>
                        <TextField name="strike" label="Strike Price" type="number" value={request.strike} onChange={handleChange} fullWidth />
                    </Grid>
                    <Grid item xs={12} sm={6}>
                        <TextField name="time_to_expiry" label="Time to Expiry (years)" type="number" value={request.time_to_expiry} onChange={handleChange} fullWidth />
                    </Grid>
                    <Grid item xs={12} sm={6}>
                        <TextField name="rate" label="Risk-free Rate" type="number" value={request.rate} onChange={handleChange} fullWidth />
                    </Grid>
                    <Grid item xs={12} sm={6}>
                        <TextField name="volatility" label="Volatility" type="number" value={request.volatility} onChange={handleChange} fullWidth />
                    </Grid>
                    <Grid item xs={12}>
                        <Button variant="contained" onClick={handleFetchPrice} disabled={loading}>
                            {loading ? <CircularProgress size={24} /> : 'Calculate Price'}
                        </Button>
                    </Grid>
                    {price !== null && (
                        <Grid item xs={12}>
                            <Typography variant="h6">Calculated Price: {price.toFixed(4)}</Typography>
                        </Grid>
                    )}
                    {error && (
                        <Grid item xs={12}>
                            <Typography color="error">{error}</Typography>
                        </Grid>
                    )}
                </Grid>
            </Paper>
        </Container>
    );
};

export default Analysis;