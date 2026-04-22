# How to Share Proteome-X

Since Proteome-X is a full-stack application (React frontend + FastAPI backend), you have two main ways to share it with others.

## 1. Quick Demo (Live from your Laptop)
If you want to show it to someone **right now** while it's running on your computer:
1. Install **ngrok**: `brew install ngrok`
2. Run your app using `./start.sh`.
3. In a new terminal, run: `ngrok http 5173`
4. ngrok will give you a public URL (e.g., `https://xyz.ngrok-free.app`) that you can send to anyone. 
   - *Note: For this to work, you may also need to expose the backend port 8000.*

---

## 2. Professional Deployment (Always Online)
This is the "Gold Standard" for your resume.

### Path A: Frontend (Vercel/Netlify) - FREE
1. Go to [Vercel](https://vercel.com).
2. Connect your GitHub repository.
3. Select the `proteome-x/frontend` folder.
4. Set the Environment Variable `VITE_API_URL` to your backend URL.
5. Hit **Deploy**. You'll get a URL like `proteome-x.vercel.app`.

### Path B: Backend (Render/Railway) - FREE/LOW COST
1. Go to [Render](https://render.com) or [Railway](https://railway.app).
2. Create a "Web Service".
3. Point it to the `proteome-x/backend` folder.
4. It will automatically detect the `Dockerfile` and deploy your API.

---

## 3. GitHub Showcase (Best for Recruiters)
1. Push this code to a public GitHub repository.
2. In your **README.md**, include the screenshot you took earlier.
3. Mention the tech stack: **GNN (PyTorch Geometric), FastAPI, React, Tailwind, ONNX**.
4. Link to the live Vercel URL (from Path A).
