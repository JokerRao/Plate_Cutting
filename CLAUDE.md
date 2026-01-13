# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A full-stack plate cutting optimization system that generates intelligent cutting plans for material optimization. The system uses advanced 2D bin packing algorithms to minimize waste and provides visual representations of cutting layouts.

## Development Commands

### Frontend (Next.js)
```bash
cd frontend
npm install              # Install dependencies
npm run dev             # Start dev server with Turbopack (http://localhost:3000)
npm run build           # Build for production
npm start               # Start production server
npm run lint            # Run ESLint
```

### Backend (FastAPI)
```bash
cd backend
python -m venv venv     # Create virtual environment (first time only)
source venv/bin/activate  # Activate venv (macOS/Linux)
pip install -r requirements.txt  # Install dependencies
python run.py           # Start server (http://localhost:8000)
```

### Testing
```bash
cd backend
pytest                  # Run all tests
pytest tests/test_api.py  # Run specific test file
pytest -v               # Verbose output
pytest --cov            # Run with coverage report
```

## Architecture

### Backend Structure

The backend follows a modular FastAPI architecture:

- **api.py**: Main FastAPI application setup, middleware configuration (CORS, rate limiting, compression), and API endpoint definitions
- **main.py**: Core optimization algorithm using rectpack library for 2D bin packing. Contains:
  - `CuttingConfig`, `SmallPlate`, `Cut` dataclasses
  - `Rectangle` class for geometric operations
  - `optimize_cutting()` function - main entry point for optimization
- **config.py**: Centralized configuration using Pydantic Settings, loads from `.env.local`
- **run.py**: Uvicorn server launcher with production-ready settings
- **app/**: Modular application structure (currently being migrated to)
  - `api/routes/`: API route handlers (health, optimization)
  - `core/`: Core utilities (constants, validation)
  - `models/`: Pydantic schemas
  - `services/`: Business logic (optimization service)

### Frontend Structure

Next.js 14 App Router architecture:

- **src/app/**: Next.js pages using App Router
  - `project/`: Project list and management
  - `project/[id]/`: Project detail page with data input
  - `layout/[id]/`: Cutting plan visualization
  - `layout/[id]/[page]/`: Paginated layout views
  - `login/`: Authentication pages
- **src/components/**: Reusable React components
- **src/config/api.ts**: API endpoint configuration
- **src/utils/supabaseClient.ts**: Supabase client initialization

### Key Technologies

- **Optimization**: Uses `rectpack` library for 2D rectangle packing algorithm
- **Database**: Supabase for real-time data storage and authentication
- **API**: FastAPI with rate limiting (slowapi), CORS, GZip compression
- **Frontend**: Next.js 15 with React 19, TypeScript, Tailwind CSS 4

## Configuration

### Environment Variables

Both frontend and backend require `.env.local` files:

**Backend** (`backend/.env.local`):
```env
HOST=127.0.0.1
PORT=8000
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

**Frontend** (`frontend/.env.local`):
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_key
```

### Server Configuration

Backend server settings in `config.py`:
- Default port: 8000
- Rate limiting: 5 requests/second
- Timeout: 300 seconds
- Workers: 1 (increase for production)
- CORS origins configured for localhost:3000 and production domains

## Data Flow

1. **User Input**: Frontend collects project data (plates, orders, cutting parameters)
2. **API Request**: POST to `/optimize` endpoint with:
   - `plates`: Available large plates (dimensions, quantities)
   - `orders`: Required pieces (dimensions, quantities)
   - `others`: Stock plates (optional)
   - `saw_blade`: Blade thickness for cutting gaps
   - `optimization`: Mode (0=normal, 1=optimized)
3. **Optimization**: Backend runs 2D bin packing algorithm
4. **Response**: Returns cutting plans with:
   - Layout coordinates for each piece
   - Utilization rates per plate
   - Total statistics (pieces placed, plates used)
5. **Visualization**: Frontend renders cutting layouts with SVG

## Important Constraints

- All dimensions (length, width, quantity) must be positive integers
- Saw blade thickness must be > 0 (supports decimals)
- Plate and piece dimensions must be larger than saw blade thickness
- The optimization algorithm accounts for blade thickness between cuts

## API Endpoints

- `GET /`: Redirect to API documentation
- `GET /health`: Health check endpoint
- `POST /optimize`: Main optimization endpoint
- `GET /docs`: Swagger UI documentation
- `GET /redoc`: ReDoc documentation

## Common Development Patterns

### Adding New API Endpoints

1. Define Pydantic models in `api.py` or `app/models/schemas.py`
2. Add route handler in `app/api/routes/`
3. Import and include router in `api.py`
4. Update frontend `src/config/api.ts` with new endpoint

### Modifying Optimization Algorithm

The core algorithm is in `backend/main.py`:
- `optimize_cutting()`: Main function that orchestrates the optimization
- Uses `rectpack.newPacker()` for bin packing
- Handles rotation, sorting, and placement strategies
- Returns structured cutting plans with coordinates

### Frontend Data Management

- Uses Supabase client for real-time database operations
- Project data structure includes: plates, orders, others, cutted (results)
- State management through React hooks and Supabase subscriptions
