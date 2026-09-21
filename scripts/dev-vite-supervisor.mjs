import { spawn } from 'node:child_process';
import { resolve } from 'node:path';

const viteBin = resolve(process.cwd(), 'node_modules', 'vite', 'bin', 'vite.js');
const restartDelayMs = 750;

let child = null;
let stopping = false;
let restartTimer = null;

function startVite() {
  if (stopping) return;

  child = spawn(process.execPath, [viteBin, '--host', '0.0.0.0', '--port', '1420', '--strictPort'], {
    stdio: 'inherit',
    env: process.env,
  });

  child.once('error', (error) => {
    console.error('[Raphael] Vite process error:', error);
  });

  child.once('exit', (code, signal) => {
    child = null;
    if (stopping) return;

    console.error(
      `[Raphael] Vite exited (code=${code ?? 'null'}, signal=${signal ?? 'none'}); restarting in ${restartDelayMs}ms`,
    );
    restartTimer = setTimeout(() => {
      restartTimer = null;
      startVite();
    }, restartDelayMs);
  });
}

function stop() {
  if (stopping) return;
  stopping = true;

  if (restartTimer) {
    clearTimeout(restartTimer);
    restartTimer = null;
  }

  if (child && !child.killed) {
    child.kill();
  }
}

process.once('SIGINT', stop);
process.once('SIGTERM', stop);
process.once('SIGHUP', stop);

startVite();
