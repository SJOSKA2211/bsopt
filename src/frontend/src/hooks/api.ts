import { useState, useEffect } from 'react';
import { useQuery, useMutation, gql, ApolloClient, NormalizedCacheObject, ApolloCache, QueryResult, MutationFunctionOptions, OperationVariables } from '@apollo/client'; // Import necessary Apollo Client types
import { DocumentNode } from 'graphql'; // For query/mutation types

// --- GraphQL Client Setup ---
// Assuming apolloClient is configured and available, e.g., imported from './apolloClient'
// If not, this would need to be set up.
import { apolloClient } from './apolloClient'; 

// --- GraphQL Fetching Hook ---
// Replaces useFetchData with a hook that uses Apollo Client's useQuery
// This hook is a wrapper around useQuery for consistency, but direct use is also fine.
export function useFetchDataGQL<T>(query: DocumentNode, options?: any) {
    const { data, loading, error, refetch } = useQuery(query, options);

    // Process Apollo's data structure to a simpler format if needed.
    // Apollo's data is typically nested, e.g., data.portfolios.
    // This extraction assumes a single top-level key for the query result.
    const processedData = data ? data[Object.keys(data)[0]] : null;

    return { 
        data: processedData as T | null, 
        loading, 
        error: error ? error.message : null, 
        refetch 
    };
}

// --- GraphQL Mutating Hook ---
// Replaces useMutateData with a hook that uses Apollo Client's useMutation
export function useMutateDataGQL<T>(mutation: DocumentNode, options?: any) {
    const [mutate, { data, loading, error }] = useMutation(mutation, options);

    const mutateAction = async (mutationOptions?: { variables?: any }) => {
        try {
            const response = await mutate(mutationOptions);
            return response.data; // Return the data from the mutation response
        } catch (err: any) {
            console.error(`Mutation error: ${err.message}`);
            throw err; // Re-throw to allow calling component to handle
        }
    };

    return { mutate: mutateAction, data, loading, error };
}

// --- Memory Saving Functions (Example) ---
// These are examples based on the prompt's mention of save_memory.
// You'll need to ensure these functions are correctly implemented and accessible.
export const saveUserPreference = async (key: string, value: any) => {
    try {
        // Assuming save_memory is globally available or imported correctly
        // Note: This part might need a proper API call or client interaction if save_memory is not a direct function.
        // await save_memory({ fact: `User preference: ${key}=${value}`, scope: 'project' });
        console.log(`Simulating save user preference: ${key}=${value}`); // Placeholder
    } catch (error) {
        logger.error(`Failed to save user preference ${key}: ${error}`);
    }
};

// --- Deprecated REST Hooks ---
// These are no longer the primary way to interact with the API.
// They are kept here for reference or potential fallback, but ideally removed.
// export function useFetchData<T>(endpoint: string, options?: RequestInit) { ... }
// export function useMutateData<T>(endpoint: string, method: 'POST' | 'PUT' | 'DELETE' | 'PATCH', options?: FetchOptions) { ... }

// Note: Need to ensure proper error handling and loading states are managed in components using these hooks.
