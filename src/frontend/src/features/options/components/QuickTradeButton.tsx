// src/frontend/src/features/options/components/QuickTradeButton.tsx
import React, { useState } from 'react';
import Button from '@mui/material/Button';
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogContentText from '@mui/material/DialogContentText';
import DialogActions from '@mui/material/DialogActions';
import Snackbar from '@mui/material/Snackbar';
import Alert from '@mui/material/Alert';
import CircularProgress from '@mui/material/CircularProgress';

interface OptionData {
  id: string;
  strike: number;
  [key: string]: string | number | boolean | null | undefined | unknown;
}

interface QuickTradeButtonProps {
  option: OptionData;
  type: 'call' | 'put';
  action: 'buy' | 'sell';
}

export const QuickTradeButton: React.FC<QuickTradeButtonProps> = React.memo(({ option, type, action }) => {
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [snackbar, setSnackbar] = useState<{ open: boolean; message: string; severity: 'success' | 'error' }>({
    open: false,
    message: '',
    severity: 'success',
  });

  const handleClickOpen = React.useCallback(() => {
    setOpen(true);
  }, []);

  const handleClose = React.useCallback(() => {
    setOpen(false);
  }, []);

  const handleSnackbarClose = React.useCallback(() => {
    setSnackbar((prev) => ({ ...prev, open: false }));
  }, []);

  const handleConfirm = React.useCallback(() => {
    setLoading(true);
    fetch('/api/v1/trades/execute', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ optionId: option.id, type, action, amount: 1 }),
    })
      .then(response => {
        if (!response.ok) throw new Error('Trade failed');
        return response.json();
      })
      .then(data => {
        setSnackbar({ open: true, message: data.message || 'Trade executed successfully', severity: 'success' });
        setOpen(false);
      })
      .catch(error => {
        setSnackbar({ open: true, message: error.message || 'Trade execution failed', severity: 'error' });
      })
      .finally(() => {
        setLoading(false);
      });
  }, [option.id, type, action]);

  return (
    <>
      <Button 
        variant={action === 'buy' ? 'contained' : 'outlined'} 
        color={action === 'buy' ? 'primary' : 'secondary'}
        size="small" 
        onClick={handleClickOpen}
      >
        {action.toUpperCase()}
      </Button>

      <Dialog open={open} onClose={handleClose}>
        <DialogTitle>Confirm Trade</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Are you sure you want to {action} {type} option for strike {option.strike}?
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleClose} disabled={loading}>Cancel</Button>
          <Button onClick={handleConfirm} autoFocus disabled={loading}>
            {loading ? <CircularProgress size={24} aria-label="Executing trade" /> : 'Confirm'}
          </Button>
        </DialogActions>
      </Dialog>

      <Snackbar open={snackbar.open} autoHideDuration={6000} onClose={handleSnackbarClose}>
        <Alert onClose={handleSnackbarClose} severity={snackbar.severity} sx={{ width: '100%' }}>
          {snackbar.message}
        </Alert>
      </Snackbar>
    </>
  );
});
