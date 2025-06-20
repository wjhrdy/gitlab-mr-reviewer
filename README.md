# GitLab MR Reviewer

A webhook service for automated GitLab Merge Request (MR) reviews using AI/LLM analysis.

---

## Architecture

- **FastAPI Webhook Service**: Receives GitLab events (`/webhook/gitlab`) and orchestrates analysis
- **Modular Pipeline**: Context generators → Analyzers (LLM/rule-based) → Notifiers
- **Repository-Specific Prompts**: Centrally managed, customizable per repo pattern
- **Automatic Cleanup**: Temporary directories (git clones) are always cleaned up

### Key Components
- `framework/webhook_service.py` - FastAPI entrypoint and event handlers
- `framework/config/prompts.py` - Repository-specific prompt configuration
- `framework/pipeline.py` - Pipeline orchestration
- `framework/context/` - Context generators (git clone, dbt parse, etc.)
- `framework/analysis/` - LLM analyzers (Gemini/VertexAI)
- `framework/notification/` - GitLab MR comment posting

---

## Quick Start

### Prerequisites
- Python 3.8+
- dbt-core 1.8.x and dbt-snowflake
- Google Cloud service account with VertexAI access
- GitLab API token with project access

### Installation
```bash
git clone <your-repo-url>
cd gitlab-mr-reviewer/framework
pip install -r requirements.txt
```

### Configuration
Set these environment variables:
```bash
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
GITLAB_API_TOKEN=your_gitlab_token
WEBHOOK_SECRET=your_webhook_secret
GITLAB_API_BASE_URL=https://gitlab.example.com  # Your GitLab instance
```

### Running Locally
```bash
uvicorn framework.webhook_service:app --host 0.0.0.0 --port 8000 --reload
```

### Health Check
```bash
curl http://localhost:8000/health
```

---


### Context Generator Details

- **[`GitCloneContextGenerator`](framework/context/git.py)**: Clones repository to temporary directory, provides `{repo_dir}`
- **[`DBTParseContextGenerator`](framework/context/dbt.py)**: Runs `dbt parse`, provides `{manifest_path}` and `{manifest_json}`
- **[`ManifestSummaryContextGenerator`](framework/context/manifest_minifier.py)**: Analyzes manifest for model counts and documentation completeness, provides `{manifest_summary}`
- **[`CISummaryContextGenerator`](framework/context/ci.py)**: Parses `.gitlab-ci.yml`, provides `{ci_summary}`
- **[`MRChangesContextGenerator`](framework/context/mr_changes.py)**: Extracts MR file changes, provides `{mr_changes}`

### Analysis Pipeline Flow

1. **Standard MR Analysis**: All repos get basic MR summary using webhook context
2. **Promotion Detection**: Only `dataproduct-config` repos check for data product promotions
3. **Deep Analysis**: When promotion detected, clones target dbt repo and runs comprehensive analysis

---

## Adding Your Own Repository and Prompts

### 1. Add Repository Pattern to Prompts Configuration

Edit `framework/config/prompts.py` and add your repository to the `_repo_prompts` dictionary:

```python
# In PromptManager.__init__()
self._repo_prompts: Dict[str, PromptConfig] = {
    # Existing repos...
    
    # Your new repository (exact match)
    "gitlab.example.com/your-org/your-repo": PromptConfig(
        mr_summary="Your custom MR summary prompt here...",
    ),
    
    # Or use regex pattern for multiple repos
    r"^gitlab\.example\.com/your-org/.*-api$": PromptConfig(
        mr_summary="Prompt for all API repos...",
    ),
}
```

### 2. Prompt Types and Templates

**Available prompt types:**
- `mr_summary` - Main MR analysis and review (required)
- `manifest` - dbt manifest analysis (for dbt repos)
- `ci` - CI/CD pipeline analysis (for dbt repos)
- `promotion` - Data product promotion detection (for dataproduct-config repo)

**Template variables available:**
- `{title}` - MR title
- `{description}` - MR description  
- `{source_branch}` - Source branch name
- `{target_branch}` - Target branch name
- `{commit_sha}` - Latest commit SHA
- `{diff_content}` - Full MR diff content
- `{project_id}` - GitLab project ID
- `{mr_iid}` - MR internal ID

### 3. Example Custom Prompt

```python
"gitlab.example.com/data-team/analytics-models": PromptConfig(
    mr_summary=(
        "You are a senior data analyst reviewing analytics models. "
        "Focus on data accuracy, business logic, and stakeholder impact.\n\n"
        "## Business Logic Review\n"
        "- Verify calculations align with business requirements\n"
        "- Check for potential data quality issues\n\n"
        "## Stakeholder Impact\n"
        "- Identify which dashboards/reports might be affected\n"
        "- Assess breaking changes for downstream consumers\n\n"
        "MR Diff:\n```diff\n{diff_content}\n```"
    ),
)
```

---

## Testing Your Configuration

### 1. Test Prompt Matching
```python
# In a Python shell or test script
from framework.config.prompts import prompt_manager

# Test your repo URL matches
repo_url = "gitlab.example.com/your-org/your-repo"
try:
    prompt = prompt_manager.get_prompt(repo_url, "mr_summary")
    print("✅ Prompt found:", prompt[:100] + "...")
except KeyError as e:
    print("❌ No prompt found:", e)
```

### 2. Test with Sample Webhook Payload

Create a test payload file `test_payload.json`:
```json
{
  "object_kind": "merge_request",
  "project": {
    "id": 12345,
    "web_url": "https://gitlab.example.com/your-org/your-repo"
  },
  "object_attributes": {
    "iid": 123,
    "title": "Test MR",
    "description": "Test description",
    "source_branch": "feature/test",
    "target_branch": "main",
    "action": "open"
  }
}
```

Test locally:
```bash
curl -X POST http://localhost:8000/webhook/gitlab \
  -H "Content-Type: application/json" \
  -H "X-Gitlab-Event: Merge Request Hook" \
  -H "X-Gitlab-Token: your_webhook_secret" \
  -d @test_payload.json
```

### 3. Debug Prompt Resolution

Add logging to see which prompts are being used:
```python
# Temporarily add to your webhook handler
logger.info(f"Using prompt for {repo_url}: {prompt[:100]}...")
```

---

## Docker Deployment

### Build Image
```bash
docker build -t gitlab-mr-reviewer .
```

### Run Container
```bash
docker run -p 8000:8000 \
  -v /path/to/service-account.json:/app/service-account.json:ro \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/service-account.json \
  -e GITLAB_API_TOKEN=your_token \
  -e WEBHOOK_SECRET=your_secret \
  gitlab-mr-reviewer
```

---

## OpenShift Deployment

### Create Secrets
```bash
# Google Cloud credentials
oc create secret generic gcp-service-account \
  --from-file=service-account.json=/path/to/service-account.json

# GitLab token and webhook secret
oc create secret generic gitlab-secrets \
  --from-literal=GITLAB_API_TOKEN=your_token \
  --from-literal=WEBHOOK_SECRET=your_secret
```

### Deploy Application
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gitlab-mr-reviewer
spec:
  replicas: 1
  selector:
    matchLabels:
      app: gitlab-mr-reviewer
  template:
    metadata:
      labels:
        app: gitlab-mr-reviewer
    spec:
      containers:
      - name: gitlab-mr-reviewer
        image: gitlab-mr-reviewer:latest
        ports:
        - containerPort: 8000
        env:
        - name: GOOGLE_APPLICATION_CREDENTIALS
          value: /app/service-account.json
        - name: GITLAB_API_TOKEN
          valueFrom:
            secretKeyRef:
              name: gitlab-secrets
              key: GITLAB_API_TOKEN
        - name: WEBHOOK_SECRET
          valueFrom:
            secretKeyRef:
              name: gitlab-secrets
              key: WEBHOOK_SECRET
        volumeMounts:
        - name: gcp-creds
          mountPath: /app/service-account.json
          subPath: service-account.json
        - name: tmp-volume
          mountPath: /tmp
      volumes:
      - name: gcp-creds
        secret:
          secretName: gcp-service-account
      - name: tmp-volume
        emptyDir: {}
```

### Create Route
```bash
oc expose service gitlab-mr-reviewer
oc patch route gitlab-mr-reviewer -p '{"spec":{"tls":{"termination":"edge"}}}'
```

---

## GitLab Webhook Configuration

1. Go to your GitLab project → Settings → Webhooks
2. Add webhook URL: `https://your-app-url/webhook/gitlab`
3. Set secret token (same as `WEBHOOK_SECRET`)
4. Enable triggers:
   - ✅ Merge request events
   - ✅ Comments (for `/analyze-promotion` commands)
5. Test the webhook

---

### Adding New Components
- **Context Generators**: Add to `framework/context/`
- **Analyzers**: Add to `framework/analysis/`
- **Notifiers**: Add to `framework/notification/`
- Follow the base class interfaces in `framework/interfaces.py`

---

## License

MIT License 