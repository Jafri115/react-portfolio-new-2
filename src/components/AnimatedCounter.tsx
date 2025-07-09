import { useEffect, useState, useRef } from 'react';

interface AnimatedCounterProps {
  value: number;
  decimal?: boolean;
}

const AnimatedCounter = ({ value, decimal = false }: AnimatedCounterProps) => {
  const [count, setCount] = useState(0);
  const ref = useRef<HTMLSpanElement | null>(null);

  useEffect(() => {
    let start = 0;
    const duration = 2000;
    const frameRate = 16;
    const totalFrames = duration / frameRate;
    const increment = (value - start) / totalFrames;

    let current = start;
    const timer = setInterval(() => {
      current += increment;
      if (current >= value) {
        clearInterval(timer);
        setCount(value);
      } else {
        setCount(current);
      }
    }, frameRate);

    return () => clearInterval(timer);
  }, [value]);

  return (
    <span ref={ref}>
      {decimal ? count.toFixed(1) : Math.floor(count)}
    </span>
  );
};

export default AnimatedCounter;
