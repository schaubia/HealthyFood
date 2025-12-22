# 🚀 Deployment Guide

This guide covers various deployment options for the Food Health Analyzer application.

## Table of Contents

- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
  - [Heroku](#heroku)
  - [Google Cloud Platform](#google-cloud-platform)
  - [AWS](#aws)
  - [Hugging Face Spaces](#hugging-face-spaces)
- [Production Considerations](#production-considerations)

## Local Development

### Quick Start

**Unix/Linux/macOS:**
```bash
./setup.sh
source venv/bin/activate
python app.py
```

**Windows:**
```cmd
setup.bat
venv\Scripts\activate
python app.py
```

### Manual Setup

1. Create virtual environment:
```bash
python -m venv venv
```

2. Activate it:
- Windows: `venv\Scripts\activate`
- Unix: `source venv/bin/activate`

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

## Docker Deployment

### Using Docker Compose (Recommended)

1. **Set environment variables** (optional):
```bash
export USDA_API_KEY=your_api_key_here
```

2. **Build and run**:
```bash
docker-compose up -d
```

3. **Access the app** at `http://localhost:7860`

4. **Stop the app**:
```bash
docker-compose down
```

### Using Docker directly

1. **Build the image**:
```bash
docker build -t food-health-analyzer .
```

2. **Run the container**:
```bash
docker run -p 7860:7860 -e USDA_API_KEY=your_key food-health-analyzer
```

## Cloud Deployment

### Hugging Face Spaces (Easiest)

Hugging Face Spaces provides free hosting for Gradio apps!

1. **Create a new Space** at [huggingface.co/spaces](https://huggingface.co/spaces)

2. **Select Gradio as the SDK**

3. **Upload your files**:
   - `app.py`
   - `requirements.txt`
   - `README.md` (optional)

4. **Set environment variables** in Space settings:
   - `USDA_API_KEY`: your_api_key

5. **The app will automatically deploy!**

Your app will be available at: `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`

### Heroku

1. **Install Heroku CLI** and login:
```bash
heroku login
```

2. **Create a new Heroku app**:
```bash
heroku create your-app-name
```

3. **Create a `Procfile`**:
```
web: python app.py
```

4. **Modify app.py** to use PORT from environment:
```python
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
```

5. **Set environment variables**:
```bash
heroku config:set USDA_API_KEY=your_api_key
```

6. **Deploy**:
```bash
git push heroku main
```

7. **Open your app**:
```bash
heroku open
```

### Google Cloud Platform (GCP)

#### Using Cloud Run

1. **Install Google Cloud SDK**

2. **Build the container**:
```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/food-analyzer
```

3. **Deploy to Cloud Run**:
```bash
gcloud run deploy food-analyzer \
  --image gcr.io/YOUR_PROJECT_ID/food-analyzer \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars USDA_API_KEY=your_api_key
```

#### Using App Engine

1. **Create `app.yaml`**:
```yaml
runtime: python311
entrypoint: python app.py

env_variables:
  USDA_API_KEY: "your_api_key"

instance_class: F2
automatic_scaling:
  max_instances: 3
```

2. **Deploy**:
```bash
gcloud app deploy
```

### AWS

#### Using Elastic Beanstalk

1. **Install EB CLI**:
```bash
pip install awsebcli
```

2. **Initialize EB application**:
```bash
eb init -p python-3.11 food-analyzer
```

3. **Create environment and deploy**:
```bash
eb create food-analyzer-env
```

4. **Set environment variables**:
```bash
eb setenv USDA_API_KEY=your_api_key
```

5. **Open your app**:
```bash
eb open
```

#### Using ECS (Docker)

1. **Push image to ECR**:
```bash
aws ecr create-repository --repository-name food-analyzer
docker tag food-health-analyzer:latest YOUR_ECR_URI
docker push YOUR_ECR_URI
```

2. **Create ECS cluster and task definition**

3. **Deploy service**

### Railway

Railway offers easy deployment with automatic HTTPS!

1. **Install Railway CLI**:
```bash
npm i -g @railway/cli
```

2. **Login and initialize**:
```bash
railway login
railway init
```

3. **Deploy**:
```bash
railway up
```

4. **Set environment variables** in Railway dashboard:
   - `USDA_API_KEY`: your_api_key

### Render

1. **Create a new Web Service** at [render.com](https://render.com)

2. **Connect your GitHub repository**

3. **Configure the service**:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python app.py`
   - Environment Variables: Add `USDA_API_KEY`

4. **Deploy!**

## Production Considerations

### Performance Optimization

1. **Enable model caching**:
```python
@st.cache_resource
def load_model():
    return FoodHealthAnalyzer()
```

2. **Use a CDN** for static assets

3. **Implement rate limiting** for API calls

4. **Add request caching** for common queries

### Security

1. **Use environment variables** for sensitive data:
```python
import os
USDA_API_KEY = os.environ.get('USDA_API_KEY')
```

2. **Enable HTTPS** (most platforms do this automatically)

3. **Implement CORS** if needed:
```python
demo.launch(
    allowed_paths=["/"],
    allowed_origins=["https://yourdomain.com"]
)
```

4. **Add rate limiting** to prevent abuse

### Monitoring

1. **Add logging**:
```python
import logging
logging.basicConfig(level=logging.INFO)
```

2. **Use application monitoring tools**:
   - Sentry for error tracking
   - Google Analytics for usage tracking
   - New Relic or Datadog for performance monitoring

3. **Set up health checks**:
```python
@app.route('/health')
def health_check():
    return {'status': 'healthy'}
```

### Scaling

1. **Horizontal scaling**: Deploy multiple instances behind a load balancer

2. **Use a queue system** for batch processing:
   - Redis Queue (RQ)
   - Celery
   - AWS SQS

3. **Implement caching**:
   - Redis for API responses
   - CDN for static content

### Cost Optimization

1. **Free Tiers**:
   - Hugging Face Spaces: Free for public apps
   - Railway: $5 free credit monthly
   - Render: Free tier available
   - Heroku: Free tier (with limitations)

2. **Paid Options** (Recommended for production):
   - GCP Cloud Run: Pay per request
   - AWS Lambda: Pay per invocation
   - Digital Ocean App Platform: $5/month starter

3. **Cost Monitoring**:
   - Set up billing alerts
   - Monitor API usage
   - Use serverless for variable load

## Troubleshooting

### Common Issues

**Port binding issues**:
```python
# Ensure the app binds to 0.0.0.0
demo.launch(server_name="0.0.0.0", server_port=7860)
```

**Memory issues**:
- Use smaller model variants
- Implement lazy loading
- Add memory limits in Docker

**Timeout issues**:
- Increase timeout settings
- Use asynchronous processing
- Implement request queuing

## Environment Variables

Required:
- `USDA_API_KEY`: Your USDA FoodData Central API key (optional, defaults to DEMO_KEY)

Optional:
- `PORT`: Port to run the app (defaults to 7860)
- `GRADIO_SERVER_NAME`: Server name (defaults to 0.0.0.0)

## Health Check Endpoints

Add these for production monitoring:

```python
@app.route('/health')
def health():
    return {'status': 'healthy', 'version': '1.0.0'}

@app.route('/ready')
def ready():
    # Check if model is loaded
    return {'status': 'ready', 'model_loaded': True}
```

## Backup and Recovery

1. **Database backups** (if using one)
2. **Model weights** backup
3. **Configuration** version control
4. **Disaster recovery plan**

---

**Need help?** Open an issue on GitHub or check our documentation!
