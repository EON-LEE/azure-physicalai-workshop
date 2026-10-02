import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { AppEntry } from './AppEntry';
import './styles.css';

const root = document.getElementById('root');
if (!root) throw new Error('Factory Console root element is missing.');

createRoot(root).render(
  <StrictMode>
    <AppEntry />
  </StrictMode>,
);
