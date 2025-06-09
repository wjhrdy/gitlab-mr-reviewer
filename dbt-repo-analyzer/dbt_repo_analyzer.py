#!/usr/bin/env python3
"""
GitLab Webhook Handler for Data Product Promotions
Processes MR changes and analyzes dbt manifest files using Gemini API
"""

import os
import json
import tempfile
import shutil
import subprocess
import hmac
import hashlib
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import asyncio
import logging
from dataclasses import dataclass

import httpx
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, Header
from pydantic import BaseModel
import google.generativeai as genai

# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Enable debug logging for httpx if debug mode
if LOG_LEVEL == "DEBUG":
    logging.getLogger("httpx").setLevel(logging.DEBUG)

# Configuration
GITLAB_TOKEN = os.getenv("GITLAB_API_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GITLAB_WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")
GITLAB_BASE_URL = "https://gitlab.cee.redhat.com"

if not GITLAB_TOKEN or not GEMINI_API_KEY:
    raise ValueError("GITLAB_TOKEN and GEMINI_API_KEY environment variables are required")

if not GITLAB_WEBHOOK_SECRET:
    logger.warning("GITLAB_WEBHOOK_SECRET not set - webhook validation disabled (not recommended for production)")

# Configure Gemini AI
genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="Data Product Promotion Handler")


@dataclass
class PromotionInfo:
    """Information about a data product promotion"""
    product_name: str
    product_type: str  # 'source' or 'aggregate'
    environment: str   # 'prod' or 'pre-prod'
    dbt_repo_url: str
    mr_iid: int
    project_id: int


class WebhookPayload(BaseModel):
    """GitLab webhook payload model"""
    object_kind: str
    project: Dict
    object_attributes: Dict
    changes: Optional[Dict] = None


class GitLabAPI:
    """GitLab API client"""
    
    def __init__(self, token: str, base_url: str):
        self.token = token
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
    
    async def get_mr_changes(self, project_id: int, mr_iid: int) -> List[Dict]:
        """Get MR file changes"""
        url = f"{self.base_url}/api/v4/projects/{project_id}/merge_requests/{mr_iid}/changes"
        logger.debug(f"Fetching MR changes from: {url}")
        
        async with httpx.AsyncClient(verify=False) as client:
            response = await client.get(url, headers=self.headers)
            logger.debug(f"GitLab API response status: {response.status_code}")
            
            if response.status_code != 200:
                logger.error(f"Failed to get MR changes: {response.text}")
                raise HTTPException(status_code=response.status_code, 
                                  detail=f"Failed to get MR changes: {response.text}")
            
            changes = response.json().get("changes", [])
            logger.debug(f"Found {len(changes)} file changes")
            return changes
    
    async def create_or_update_comment(self, project_id: int, mr_iid: int, content: str):
        """Create or update a comment on the MR"""
        # First, try to find existing bot comment
        existing_comment = await self._find_bot_comment(project_id, mr_iid)
        
        if existing_comment:
            logger.debug(f"Found existing comment with ID: {existing_comment['id']}")
            await self._update_comment(project_id, mr_iid, existing_comment["id"], content)
        else:
            logger.debug("No existing comment found, creating new one")
            await self._create_comment(project_id, mr_iid, content)
    
    async def _find_bot_comment(self, project_id: int, mr_iid: int) -> Optional[Dict]:
        """Find existing bot comment"""
        url = f"{self.base_url}/api/v4/projects/{project_id}/merge_requests/{mr_iid}/notes"
        
        async with httpx.AsyncClient(verify=False) as client:
            response = await client.get(url, headers=self.headers)
            if response.status_code == 200:
                notes = response.json()
                for note in notes:
                    if note.get("body", "").startswith("## 🚀 Data Product Promotion Analysis"):
                        return note
        return None
    
    async def _create_comment(self, project_id: int, mr_iid: int, content: str):
        """Create a new comment"""
        url = f"{self.base_url}/api/v4/projects/{project_id}/merge_requests/{mr_iid}/notes"
        data = {"body": content}
        
        async with httpx.AsyncClient(verify=False) as client:
            response = await client.post(url, headers=self.headers, json=data)
            if response.status_code not in [200, 201]:
                logger.error(f"Failed to create comment: {response.text}")
    
    async def _update_comment(self, project_id: int, mr_iid: int, note_id: int, content: str):
        """Update existing comment"""
        url = f"{self.base_url}/api/v4/projects/{project_id}/merge_requests/{mr_iid}/notes/{note_id}"
        data = {"body": content}
        
        async with httpx.AsyncClient(verify=False) as client:
            response = await client.put(url, headers=self.headers, json=data)
            if response.status_code not in [200, 201]:
                logger.error(f"Failed to update comment: {response.text}")
                logger.debug(f"Update comment URL: {url}")
                logger.debug(f"Response status: {response.status_code}")
                # If update fails, try creating a new comment instead
                logger.info("Update failed, creating new comment instead")
                await self._create_comment(project_id, mr_iid, content)
            else:
                logger.debug(f"Successfully updated comment {note_id}")


class GeminiAnalyzer:
    """Gemini AI analyzer for MR changes and dbt manifests"""
    
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    async def analyze_mr_for_promotion(self, changes: List[Dict]) -> Optional[PromotionInfo]:
        """Analyze MR changes to detect data product promotions"""
        
        logger.debug(f"Analyzing {len(changes)} changes for promotion patterns")
        
        # Extract file paths from changes
        file_changes = []
        for change in changes:
            if change.get("new_file") or change.get("renamed_file"):
                file_path = change.get("new_path", change.get("old_path", ""))
                file_changes.append({
                    "path": file_path,
                    "action": "added" if change.get("new_file") else "modified"
                })
                logger.debug(f"Found file change: {file_path}")
        
        if not file_changes:
            logger.debug("No relevant file changes found")
            return None
        
        prompt = f"""
        Analyze these GitLab MR file changes to determine if they represent a data product promotion:

        File changes:
        {json.dumps(file_changes, indent=2)}

        Rules for promotion detection:
        1. Look for additions of 'product.yaml' files to 'prod' or 'pre-prod' directories
        2. The pattern should be: dataproducts/[source|aggregate]/[product_name]/[prod|pre-prod]/product.yaml
        3. Extract the product name from the parent directory of the prod/pre-prod folder
        4. Determine if it's source-aligned or aggregate based on the path

        Respond with JSON only:
        {{
            "is_promotion": boolean,
            "product_name": "string or null",
            "product_type": "source or aggregate or null", 
            "environment": "prod or pre-prod or null",
            "confidence": "high/medium/low"
        }}
        """
        
        try:
            logger.debug("Sending analysis request to Gemini API")
            response = await asyncio.to_thread(self.model.generate_content, prompt)
            logger.debug(f"Gemini API response: {response.text[:200]}...")
            
            # Extract JSON from markdown code blocks if present
            response_text = response.text.strip()
            if response_text.startswith("```json"):
                # Remove ```json and ``` markers
                response_text = response_text[7:]  # Remove ```json
                if response_text.endswith("```"):
                    response_text = response_text[:-3]  # Remove ```
                response_text = response_text.strip()
            elif response_text.startswith("```"):
                # Handle generic ``` blocks
                lines = response_text.split('\n')
                if len(lines) > 2:
                    response_text = '\n'.join(lines[1:-1])  # Remove first and last lines
                response_text = response_text.strip()
            
            logger.debug(f"Cleaned response text: {response_text}")
            result = json.loads(response_text)
            logger.debug(f"Parsed Gemini result: {result}")
            
            if result.get("is_promotion") and result.get("confidence") in ["high", "medium"]:
                # Construct dbt repo URL based on product type
                product_type = result["product_type"]
                product_name = result["product_name"]
                
                if product_type == "source":
                    # source-aligned: /source-aligned/product/product-dbt
                    dbt_repo_url = f"{GITLAB_BASE_URL}/dataverse/data-products/source-aligned/{product_name}/{product_name}-dbt"
                elif product_type == "aggregate":
                    # aggregate: /aggregate/product/product-dbt (NOT aggregate-aligned)
                    dbt_repo_url = f"{GITLAB_BASE_URL}/dataverse/data-products/aggregate/{product_name}/{product_name}-dbt"
                else:
                    logger.warning(f"Unknown product type: {product_type}")
                    dbt_repo_url = f"{GITLAB_BASE_URL}/dataverse/data-products/{product_type}/{product_name}/{product_name}-dbt"
                
                logger.info(f"Detected promotion: {product_name} ({product_type}) to {result['environment']}")
                logger.debug(f"dbt repo URL: {dbt_repo_url}")
                
                return PromotionInfo(
                    product_name=product_name,
                    product_type=product_type,
                    environment=result["environment"],
                    dbt_repo_url=dbt_repo_url,
                    mr_iid=0,  # Will be set by caller
                    project_id=0  # Will be set by caller
                )
            else:
                logger.debug(f"No promotion detected. is_promotion: {result.get('is_promotion')}, confidence: {result.get('confidence')}")
        except Exception as e:
            logger.error(f"Error analyzing MR with Gemini: {e}")
            logger.debug(f"Full error details: {e}", exc_info=True)
        
        return None
    
    async def analyze_dbt_manifest(self, manifest_path: str) -> str:
        """Analyze dbt manifest.json and generate summary"""
        
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            # Extract key information
            nodes = manifest.get("nodes", {})
            tests = manifest.get("tests", {})
            
            # Analyze models by schema
            models_by_schema = {}
            marts_models = {}
            test_results = {"passed": 0, "failed": 0, "total": 0}
            
            for node_id, node in nodes.items():
                if node.get("resource_type") == "model":
                    schema = node.get("schema", "unknown")
                    if schema not in models_by_schema:
                        models_by_schema[schema] = 0
                    models_by_schema[schema] += 1
                    
                    # Track marts models specifically for metadata analysis
                    if "marts" in schema.lower():
                        marts_models[node.get("name", node_id)] = {
                            "description": node.get("description", ""),
                            "columns": node.get("columns", {})
                        }
                
                elif node.get("resource_type") == "test":
                    test_results["total"] += 1
                    # Note: We can't determine pass/fail from manifest alone
            
            # Build focused summary
            prompt = f"""
            Create a crisp, focused dbt project analysis covering these specific areas:

            SCHEMAS AND MODEL COUNTS:
            {json.dumps(models_by_schema, indent=2)}

            MARTS MODELS METADATA:
            {json.dumps(marts_models, indent=2)}

            TESTS:
            - Total tests: {test_results["total"]}

            Create a concise markdown summary with these sections only:
            1. **Test Coverage** - Total number of tests and overall assessment
            2. **Schemas & Models** - List each schema with model count
            3. **Marts Documentation** - For each model in marts schema, show:
               - Model name and description (if any)
               - Column documentation status (how many columns have descriptions)

            Keep it brief and actionable. Focus on data quality and documentation gaps.
            Do NOT wrap the response in code blocks.
            """
            
            logger.debug("Sending manifest analysis request to Gemini API")
            response = await asyncio.to_thread(self.model.generate_content, prompt)
            
            # Clean up any markdown code blocks that might be returned
            response_text = response.text.strip()
            if response_text.startswith("```") and response_text.endswith("```"):
                lines = response_text.split('\n')
                if len(lines) > 2:
                    response_text = '\n'.join(lines[1:-1])  # Remove first and last lines
                response_text = response_text.strip()
            
            logger.debug(f"Manifest analysis completed, response length: {len(response_text)}")
            return response_text
            
        except Exception as e:
            logger.error(f"Error analyzing manifest with Gemini: {e}")
            return f"❌ Error analyzing dbt manifest: {str(e)}"


class DBTRunner:
    """Handle dbt operations"""
    
    @staticmethod
    async def clone_and_parse(repo_url: str, gitlab_token: str = None) -> Tuple[bool, str, Optional[str]]:
        """Clone repo and run dbt parse, return (success, message, manifest_path)"""
        
        temp_dir = None
        try:
            # Create temporary directory in /tmp (mounted as emptyDir in OpenShift)
            temp_dir = tempfile.mkdtemp(prefix="dbt_", dir="/tmp")
            logger.info(f"Created temporary directory: {temp_dir}")
            
            # Clone repository with timeout (disable SSL verification)
            clone_cmd = ["git", "-c", "http.sslVerify=false", "clone", "--depth", "1", repo_url, temp_dir]
            
            result = await asyncio.to_thread(
                subprocess.run, 
                clone_cmd, 
                capture_output=True, 
                text=True, 
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                logger.error(f"Git clone failed: {result.stderr}")
                return False, f"Git clone failed: {result.stderr}", None
            
            logger.info(f"Successfully cloned repository to {temp_dir}")
            
            # Set up dbt environment with comprehensive SSL bypass
            env = os.environ.copy()
            env["DBT_PROFILES_DIR"] = temp_dir
            # Disable SSL verification for any operations that might need it
            env["PYTHONHTTPSVERIFY"] = "0"
            env["CURL_CA_BUNDLE"] = ""
            env["REQUESTS_CA_BUNDLE"] = ""
            env["SSL_VERIFY"] = "false"
            env["GIT_SSL_NO_VERIFY"] = "true"
            
            # Configure git SSL settings in this session
            git_configs = [
                ["git", "config", "http.sslVerify", "false"],
                ["git", "config", "http.sslCAInfo", ""],
                ["git", "config", "http.sslCAPath", ""],
                ["git", "config", "http.sslCert", ""],
                ["git", "config", "http.sslKey", ""],
                ["git", "config", "http.sslCertPasswordProtected", "false"]
            ]
            
            for config_cmd in git_configs:
                try:
                    await asyncio.to_thread(
                        subprocess.run, 
                        config_cmd, 
                        cwd=temp_dir, 
                        capture_output=True, 
                        text=True,
                        timeout=30
                    )
                    logger.debug(f"Applied git config: {' '.join(config_cmd)}")
                except Exception as e:
                    logger.debug(f"Git config failed (continuing): {e}")
            
            # Check dbt_project.yml to see what profile is expected
            dbt_project_path = os.path.join(temp_dir, "dbt_project.yml")
            expected_profile = "default"  # fallback
            
            if os.path.exists(dbt_project_path):
                try:
                    import yaml
                    with open(dbt_project_path, 'r') as f:
                        dbt_project = yaml.safe_load(f)
                        expected_profile = dbt_project.get('profile', 'default')
                        logger.debug(f"Found dbt profile in project: {expected_profile}")
                except Exception as e:
                    logger.debug(f"Could not read dbt_project.yml: {e}")
            
            # Create profiles.yml with the expected profile name (Snowflake adapter)
            profiles_content = f"""
{expected_profile}:
  outputs:
    dev:
      type: snowflake
      account: dummy_account
      user: dummy_user
      password: dummy_password
      role: dummy_role
      database: dummy_database
      warehouse: dummy_warehouse
      schema: public
      threads: 1
  target: dev

default:
  outputs:
    dev:
      type: snowflake
      account: dummy_account
      user: dummy_user
      password: dummy_password
      role: dummy_role
      database: dummy_database
      warehouse: dummy_warehouse
      schema: public
      threads: 1
  target: dev

local:
  outputs:
    dev:
      type: snowflake
      account: dummy_account
      user: dummy_user
      password: dummy_password
      role: dummy_role
      database: dummy_database
      warehouse: dummy_warehouse
      schema: public
      threads: 1
  target: dev
"""
            profiles_path = os.path.join(temp_dir, "profiles.yml")
            with open(profiles_path, 'w') as f:
                f.write(profiles_content)
            
            # Run dbt deps first to install package dependencies
            logger.info("Running dbt deps to install package dependencies...")
            deps_cmd = ["dbt", "deps", "--profiles-dir", temp_dir]
            result = await asyncio.to_thread(
                subprocess.run, 
                deps_cmd, 
                cwd=temp_dir, 
                capture_output=True, 
                text=True, 
                env=env,
                timeout=300  # 5 minute timeout for deps
            )
            
            if result.returncode != 0:
                logger.warning(f"dbt deps failed (continuing anyway): {result.stderr}")
                # Don't fail here - some projects might not need deps or have issues
                # but we can still try to parse
            else:
                logger.info("dbt deps completed successfully")
            
            # Run dbt parse with timeout
            parse_cmd = ["dbt", "parse", "--profiles-dir", temp_dir]
            logger.info("Running dbt parse...")
            result = await asyncio.to_thread(
                subprocess.run, 
                parse_cmd, 
                cwd=temp_dir, 
                capture_output=True, 
                text=True, 
                env=env,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode != 0:
                logger.error(f"dbt parse failed: {result.stderr}")
                logger.debug(f"dbt parse stdout: {result.stdout}")
                return False, f"dbt parse failed: {result.stderr}", None
            
            logger.info("dbt parse completed successfully")
            
            # Check for manifest.json
            manifest_path = os.path.join(temp_dir, "target", "manifest.json")
            if not os.path.exists(manifest_path):
                logger.error(f"manifest.json not found at {manifest_path}")
                return False, "manifest.json not found after dbt parse", None
            
            logger.info(f"Found manifest.json at {manifest_path}")
            return True, "dbt parse successful", manifest_path
            
        except subprocess.TimeoutExpired:
            logger.error("dbt operations timed out")
            return False, "dbt operations timed out", None
        except Exception as e:
            logger.error(f"Error in dbt operations: {str(e)}")
            return False, f"Error in dbt operations: {str(e)}", None
        finally:
            # Cleanup will happen after manifest analysis
            pass


async def process_promotion(promotion: PromotionInfo, gitlab_api: GitLabAPI):
    """Process a data product promotion"""
    
    logger.info(f"Processing promotion for {promotion.product_name} to {promotion.environment}")
    
    # Update MR with initial status
    initial_comment = f"""## 🚀 Data Product Promotion Analysis

**Product**: {promotion.product_name}  
**Type**: {promotion.product_type}-aligned  
**Environment**: {promotion.environment}  

Cloning dbt repository and running analysis...
"""
    
    await gitlab_api.create_or_update_comment(
        promotion.project_id, promotion.mr_iid, initial_comment
    )
    
    # Clone repo and run dbt parse (GitLab token only used for API calls)
    success, message, manifest_path = await DBTRunner.clone_and_parse(
        promotion.dbt_repo_url
    )
    
    if not success:
        error_comment = f"""## 🚀 Data Product Promotion Analysis

**Product**: {promotion.product_name}  
**Type**: {promotion.product_type}-aligned  
**Environment**: {promotion.environment}  

**Error**: {message}
"""
        await gitlab_api.create_or_update_comment(
            promotion.project_id, promotion.mr_iid, error_comment
        )
        return
    
    # Analyze manifest with Gemini
    analyzer = GeminiAnalyzer()
    manifest_summary = await analyzer.analyze_dbt_manifest(manifest_path)
    
    # Clean up temporary directory
    temp_dir = os.path.dirname(os.path.dirname(manifest_path))
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Create final comment
    final_comment = f"""## 🚀 Data Product Promotion Analysis

**Product**: {promotion.product_name}  
**Type**: {promotion.product_type}-aligned  
**Environment**: {promotion.environment}  

### dbt Project Analysis

{manifest_summary}

---
*Analysis completed by Data Product Promotion Bot*
"""
    
    await gitlab_api.create_or_update_comment(
        promotion.project_id, promotion.mr_iid, final_comment
    )


def verify_gitlab_signature(payload: bytes, signature: str, secret: str) -> bool:
    """Verify GitLab webhook signature"""
    if not secret:
        return True  # Skip validation if no secret configured
    
    expected_signature = hmac.new(
        secret.encode('utf-8'),
        payload,
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(signature, expected_signature)


@app.post("/webhook/gitlab")
async def handle_gitlab_webhook(
    request: Request, 
    background_tasks: BackgroundTasks,
    x_gitlab_token: Optional[str] = Header(None)
):
    """Handle GitLab webhook for MR events"""
    
    try:
        # Get raw payload for signature verification
        payload_bytes = await request.body()

        logger.debug(f"Received GitLab webhook payload of size {len(payload_bytes)} bytes")
        
        # Verify webhook signature if secret is configured
        if GITLAB_WEBHOOK_SECRET:
            if not x_gitlab_token:
                raise HTTPException(status_code=401, detail="Missing X-Gitlab-Token header")
            
            if not verify_gitlab_signature(payload_bytes, x_gitlab_token, GITLAB_WEBHOOK_SECRET):
                raise HTTPException(status_code=401, detail="Invalid webhook signature")
        logger.debug("Webhook signature verified successfully")
        
        # Parse JSON payload
        try:
            payload = json.loads(payload_bytes.decode('utf-8'))
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON payload")
        

        logger.debug(f"Parsed GitLab webhook payload: {payload.get('object_kind', 'unknown')} event")
        
        # Only process merge request events
        if payload.get("object_kind") != "merge_request":
            return {"status": "ignored", "reason": "not a merge request event"}
        
        logger.debug("Processing merge request event")  
        # Only process opened or updated MRs
        mr_action = payload.get("object_attributes", {}).get("action")
        if mr_action not in ["open", "update"]:
            return {"status": "ignored", "reason": f"action '{mr_action}' not relevant"}
        
        logger.debug(f"Merge request action: {mr_action}")
        
        project_id = payload["project"]["id"]
        mr_iid = payload["object_attributes"]["iid"]
        
        # Get MR changes
        gitlab_api = GitLabAPI(GITLAB_TOKEN, GITLAB_BASE_URL)
        changes = await gitlab_api.get_mr_changes(project_id, mr_iid)

        logger.debug(f"Retrieved {len(changes)} changes for MR {mr_iid} in project {project_id}")
        
        # Analyze changes with Gemini
        analyzer = GeminiAnalyzer()
        promotion = await analyzer.analyze_mr_for_promotion(changes)

        logger.debug(f"Promotion analysis result: {promotion}")
        
        if not promotion:
            return {"status": "ignored", "reason": "no data product promotion detected"}
        
        # Set MR details
        promotion.mr_iid = mr_iid
        promotion.project_id = project_id
        
        # Process promotion in background
        background_tasks.add_task(process_promotion, promotion, gitlab_api)
        
        return {
            "status": "processing",
            "product": promotion.product_name,
            "environment": promotion.environment
        }
        
    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "data-product-promotion-handler"}


if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment (OpenShift typically uses PORT env var)
    port = int(os.getenv("PORT", 8000))
    
    # Configure for production deployment
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        workers=int(os.getenv("WORKERS", 1)),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        access_log=True
    )