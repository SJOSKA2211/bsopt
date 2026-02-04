import { expect, test, vi } from 'vitest';
import { render, screen, act } from '@testing-library/react';
import React, { useState } from 'react';

test('Memoized component should not re-render when props do not change', async () => {
  const onRender = vi.fn();
  
  const MemoizedComponent = React.memo(({ data }: { data: { value: string } }) => {
    onRender();
    return <div>{data.value}</div>;
  });

  const Parent = () => {
    const [count, setCount] = useState(0);
    const [data] = useState({ value: 'fixed' });

    return (
      <div>
        <button onClick={() => setCount(count + 1)}>Increment</button>
        <MemoizedComponent data={data} />
      </div>
    );
  };

  render(<Parent />);
  
  // Initial render
  expect(onRender).toHaveBeenCalledTimes(1);

  // Trigger state change in parent
  const button = screen.getByText('Increment');
  await act(async () => {
    button.click();
  });

  // Should NOT have re-rendered MemoizedComponent
  expect(onRender).toHaveBeenCalledTimes(1);
});

test('Non-memoized component re-renders when parent re-renders', async () => {
  const onRender = vi.fn();
  
  const NonMemoizedComponent = ({ data }: { data: { value: string } }) => {
    onRender();
    return <div>{data.value}</div>;
  };

  const Parent = () => {
    const [count, setCount] = useState(0);
    const [data] = useState({ value: 'fixed' });

    return (
      <div>
        <button onClick={() => setCount(count + 1)}>Increment</button>
        <NonMemoizedComponent data={data} />
      </div>
    );
  };

  render(<Parent />);
  
  // Initial render
  expect(onRender).toHaveBeenCalledTimes(1);

  // Trigger state change in parent
  const button = screen.getByText('Increment');
  await act(async () => {
    button.click();
  });

  // SHOULD have re-rendered NonMemoizedComponent
  expect(onRender).toHaveBeenCalledTimes(2);
});
