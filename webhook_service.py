#!/usr/bin/env python3
"""
Flask webhook service with native GitLab MR summarizer implementation.
No subprocess calls - everything runs in the same process.
"""

import os
import json
import hmac
import hashlib
import sys
import requests
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional
from datetime import datetime
from flask import Flask, request, jsonify, abort
import logging
import google.generativeai as genai
import urllib3

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WEBHOOK_SECRET = os.getenv('WEBHOOK_SECRET', '')

class GitLabMRSummarizer:
    """GitLab MR Summarizer integrated into the webhook service."""
    
    def __init__(self, webhook_data: Dict):
        """Initialize summarizer with webhook data."""
        self.webhook_data = webhook_data
        
        # Extract data from webhook
        mr_data = webhook_data.get('object_attributes', {})
        project = webhook_data.get('project', {})
        
        # Set up GitLab connection info
        self.gitlab_token = os.getenv('GITLAB_API_TOKEN')
        self.project_id = str(project.get('id', ''))
        self.mr_iid = str(mr_data.get('iid', ''))
        
        # Extract GitLab server URL from project web_url
        project_web_url = project.get('web_url', '')
        project_path = project.get('path_with_namespace', '')
        self.gitlab_url = project_web_url.replace(f"/{project_path}", '') if project_path else project_web_url
        
        # Ensure HTTPS
        if self.gitlab_url.startswith('http://'):
            self.gitlab_url = self.gitlab_url.replace('http://', 'https://')
            logger.warning(f"Converted insecure URL to HTTPS: {self.gitlab_url}")
        
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        
        # Validate required parameters
        if not all([self.gitlab_token, self.project_id, self.mr_iid, self.gemini_api_key]):
            missing = []
            if not self.gitlab_token: missing.append('GITLAB_API_TOKEN')
            if not self.project_id: missing.append('project_id from webhook')
            if not self.mr_iid: missing.append('mr_iid from webhook')
            if not self.gemini_api_key: missing.append('GEMINI_API_KEY')
            raise ValueError(f"Missing required parameters: {', '.join(missing)}")
        
        # Configure Gemini
        genai.configure(api_key=self.gemini_api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # GitLab API headers
        self.headers = {
            'Authorization': f'Bearer {self.gitlab_token}',
            'Content-Type': 'application/json',
            'User-Agent': 'GitLab-MR-Summary-Bot/1.0'
        }
        
        # Configure requests session
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # SSL verification
        verify_ssl = os.getenv('VERIFY_SSL', 'false').lower() == 'true'
        self.session.verify = verify_ssl
        
        if not verify_ssl:
            logger.info("SSL verification disabled - allowing self-signed certificates")
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    def load_prompt_template(self) -> str:
        """Load prompt template from file, URL, or fallback to default."""

        # Try loading from local file
        prompt_template_path = os.getenv('PROMPT_TEMPLATE_PATH', '/app/prompt_template.txt')
        try:
            if os.path.exists(prompt_template_path):
                logger.info(f"Loading prompt template from file: {prompt_template_path}")
                with open(prompt_template_path, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            logger.warning(f"Failed to load prompt from file: {e}")
        
        # Try loading from URL first (if provided)
        prompt_template_url = os.getenv('PROMPT_TEMPLATE_URL')
        if prompt_template_url:
            try:
                logger.info(f"Loading prompt template from URL: {prompt_template_url}")
                response = requests.get(prompt_template_url, timeout=10)
                response.raise_for_status()
                return response.text
            except Exception as e:
                logger.warning(f"Failed to load prompt from URL: {e}")
        

        
        # Fallback to embedded default prompt
        logger.info("Using embedded default prompt template")
        return self.get_default_prompt_template()
    
    def get_default_prompt_template(self) -> str:
        """Return the default embedded prompt template."""
        return """Please analyze this GitLab merge request and provide a comprehensive summary.

**Merge Request Details:**
- Title: {mr_title}
- Description: {mr_description}
- Source Branch: {source_branch}
- Target Branch: {target_branch}
- Commit: {commit_sha}

**Diff Content:**
```diff
{diff_content}
```

Please provide a summary that includes the following sections:

1. **Overview**: Brief description of what this MR accomplishes.
2. **Key Changes**: List the main changes made (focus on significant modifications).
3. **Files Modified**: Summary of which files/areas were changed
4. **Impact**: Potential impact of these changes

Format the response in clear markdown with appropriate headers and bullet points.
Keep the summary concise but comprehensive - aim for 200-500 words total.
Focus on what's most important for code reviewers to understand."""
    
    def secure_api_request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Make an API request with configurable SSL verification."""
        if url.startswith('http://'):
            url = url.replace('http://', 'https://')
            logger.warning(f"Converted HTTP to HTTPS: {url.split('/')[2]}")
        
        try:
            response = self.session.request(method, url, timeout=30, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.SSLError as e:
            if self.session.verify:
                logger.error(f"SSL Error - try setting VERIFY_SSL=false for self-signed certificates: {e}")
            else:
                logger.error(f"SSL Error even with verification disabled: {e}")
            raise
        except requests.exceptions.Timeout:
            logger.error("Request timeout - GitLab API may be slow")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error: {e}")
            raise
    
    def get_mr_diff(self) -> str:
        """Fetch the diff for the merge request."""
        url = f"{self.gitlab_url}/api/v4/projects/{self.project_id}/merge_requests/{self.mr_iid}/diffs"
        
        try:
            response = self.secure_api_request('GET', url)
            diffs = response.json()
            
            # Combine all diffs into a single string
            combined_diff = ""
            for diff in diffs:
                if 'diff' in diff:
                    combined_diff += f"\n--- File: {diff.get('new_path', diff.get('old_path', 'unknown'))}\n"
                    combined_diff += diff['diff']
                    combined_diff += "\n" + "="*80 + "\n"
            
            return combined_diff
        except Exception as e:
            logger.error(f"Error fetching MR diff: {e}")
            return ""
    
    def get_mr_details(self) -> Dict:
        """Fetch merge request details."""
        url = f"{self.gitlab_url}/api/v4/projects/{self.project_id}/merge_requests/{self.mr_iid}"
        
        try:
            response = self.secure_api_request('GET', url)
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching MR details: {e}")
            return {}
    
    def generate_summary_with_gemini(self, diff_content: str, mr_details: Dict) -> str:
        """Generate a summary using Gemini API."""
        
        # Truncate diff if it's too long
        max_diff_length = int(os.getenv('MAX_DIFF_LENGTH', '50000'))
        if len(diff_content) > max_diff_length:
            diff_content = diff_content[:max_diff_length] + "\n\n[...diff truncated...]"
        
        # Get commit info
        commit_sha = self.webhook_data.get('object_attributes', {}).get('last_commit', {}).get('id', '')[:8]
        
        # Load the prompt template
        try:
            prompt_template = self.load_prompt_template()
        except Exception as e:
            logger.error(f"Error loading prompt template: {e}")
            prompt_template = self.get_default_prompt_template()
        
        # Format the prompt with actual values
        try:
            prompt = prompt_template.format(
                mr_title=mr_details.get('title', 'N/A'),
                mr_description=mr_details.get('description', 'N/A'),
                source_branch=mr_details.get('source_branch', 'N/A'),
                target_branch=mr_details.get('target_branch', 'N/A'),
                commit_sha=commit_sha,
                diff_content=diff_content
            )
        except KeyError as e:
            logger.warning(f"Missing template variable {e}, using default template")
            prompt_template = self.get_default_prompt_template()
            prompt = prompt_template.format(
                mr_title=mr_details.get('title', 'N/A'),
                mr_description=mr_details.get('description', 'N/A'),
                source_branch=mr_details.get('source_branch', 'N/A'),
                target_branch=mr_details.get('target_branch', 'N/A'),
                commit_sha=commit_sha,
                diff_content=diff_content
            )
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"❌ Error generating summary with Gemini: {str(e)}"
    
    def find_existing_summary_comment(self) -> Optional[int]:
        """Find existing AI summary comment in the MR."""
        url = f"{self.gitlab_url}/api/v4/projects/{self.project_id}/merge_requests/{self.mr_iid}/notes"
        
        try:
            response = self.secure_api_request('GET', url)
            notes = response.json()
            
            comment_header = os.getenv('COMMENT_HEADER', '## 🤖 Gemini-Generated MR Summary')
            for note in notes:
                if note.get('body', '').startswith(comment_header):
                    return note['id']
            
            return None
        except Exception as e:
            logger.error(f"Error fetching MR comments: {e}")
            return None
    
    def post_or_update_mr_comment(self, summary: str) -> bool:
        """Post a new summary comment or update existing one."""
        commit_sha = self.webhook_data.get('object_attributes', {}).get('last_commit', {}).get('id', '')[:8]
        comment_header = os.getenv('COMMENT_HEADER', '## 🤖 Gemini-Generated MR Summary')
        
        comment_body = f"""{comment_header}

{summary}

---
<details>
<summary>📋 Summary Details</summary>

- **Commit**: {commit_sha}
- **Generated**: {self.get_current_timestamp()}
- **AI Model**: Gemini 1.5 Flash

*This summary is automatically updated when new commits are pushed to the MR.*
</details>
"""
        
        # Check if we already have a summary comment
        existing_comment_id = self.find_existing_summary_comment()
        
        if existing_comment_id:
            # Update existing comment
            url = f"{self.gitlab_url}/api/v4/projects/{self.project_id}/merge_requests/{self.mr_iid}/notes/{existing_comment_id}"
            
            try:
                response = self.secure_api_request('PUT', url, json={'body': comment_body})
                logger.info(f"Successfully updated existing summary comment (ID: {existing_comment_id})")
                return True
            except Exception as e:
                logger.error(f"Error updating existing comment: {e}")
                # Fall back to creating new comment
                return self.create_new_mr_comment(comment_body)
        else:
            # Create new comment
            return self.create_new_mr_comment(comment_body)
    
    def create_new_mr_comment(self, comment_body: str) -> bool:
        """Create a new comment on the merge request."""
        url = f"{self.gitlab_url}/api/v4/projects/{self.project_id}/merge_requests/{self.mr_iid}/notes"
        
        try:
            response = self.secure_api_request('POST', url, json={'body': comment_body})
            logger.info("Successfully posted new summary comment to MR")
            return True
        except Exception as e:
            logger.error(f"Error posting new comment to MR: {e}")
            return False
    
    def get_current_timestamp(self) -> str:
        """Get current timestamp for comment updates."""
        return datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    
    def run(self) -> tuple[bool, str]:
        """Main execution function. Returns (success, message)."""
        try:
            logger.info("Starting MR summary generation...")
            
            # Get MR details
            mr_details = self.get_mr_details()
            if not mr_details:
                return False, "Failed to fetch MR details"
            
            logger.info(f"Processing MR: {mr_details.get('title', 'N/A')}")
            
            # Get diff
            diff_content = self.get_mr_diff()
            if not diff_content:
                logger.info("No diff content found")
                summary = "No changes detected in this merge request."
            else:
                logger.info(f"Diff content length: {len(diff_content)} characters")
                
                # Generate summary
                logger.info("Generating summary with Gemini...")
                summary = self.generate_summary_with_gemini(diff_content, mr_details)
            
            logger.info("Generated summary successfully")
            
            # Post or update comment in MR
            if os.getenv('POST_COMMENT_TO_MR', 'true').lower() == 'true':
                success = self.post_or_update_mr_comment(summary)
                if success:
                    return True, f"MR summary generated and posted successfully"
                else:
                    return False, f"Summary generated but failed to post comment"
            else:
                return True, f"MR summary generated successfully (comment posting disabled)"
            
        except Exception as e:
            logger.error(f"Error in MR summarizer: {str(e)}")
            return False, f"Error: {str(e)}"


def verify_signature(payload_body, signature_header):
    """Verify GitLab webhook signature."""
    if not WEBHOOK_SECRET:
        logger.warning("No webhook secret configured - skipping signature verification")
        return True
    
    if not signature_header:
        logger.error("No X-Gitlab-Token header found")
        return False
    
    # GitLab sends the token directly, not as HMAC
    return hmac.compare_digest(signature_header, WEBHOOK_SECRET)

def should_process_event(event_data):
    """Determine if we should process this webhook event."""
    object_kind = event_data.get('object_kind')
    
    if object_kind != 'merge_request':
        logger.info(f"Ignoring non-MR event: {object_kind}")
        return False
    
    # Process on MR open, update, or reopen
    action = event_data.get('object_attributes', {}).get('action')
    if action in ['open', 'update', 'reopen']:
        logger.info(f"Processing MR {action} event")
        return True
    
    logger.info(f"Ignoring MR action: {action}")
    return False

@app.route('/webhook', methods=['POST'])
def handle_webhook():
    """Handle incoming GitLab webhook."""
    try:
        # Verify signature
        signature = request.headers.get('X-Gitlab-Token')
        if not verify_signature(request.data, signature):
            logger.error("Invalid webhook signature")
            abort(401)
        
        # Parse JSON payload
        try:
            event_data = request.get_json()
            if not event_data:
                logger.error("Empty JSON payload")
                abort(400)
        except Exception as e:
            logger.error(f"Invalid JSON payload: {str(e)}")
            abort(400)
        
        # Log the event
        object_kind = event_data.get('object_kind', 'unknown')
        project_name = event_data.get('project', {}).get('name', 'unknown')
        logger.info(f"Received webhook: {object_kind} for project {project_name}")
        
        # Check if we should process this event
        if not should_process_event(event_data):
            return jsonify({'status': 'ignored', 'message': 'Event not processed'})
        
        # Process the MR event using native summarizer
        try:
            summarizer = GitLabMRSummarizer(event_data)
            success, message = summarizer.run()
            
            if success:
                return jsonify({
                    'status': 'success',
                    'message': message
                })
            else:
                return jsonify({
                    'status': 'error',
                    'message': message
                }), 500
                
        except Exception as e:
            logger.error(f"Error initializing summarizer: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'Failed to initialize summarizer: {str(e)}'
            }), 500
            
    except Exception as e:
        logger.error(f"Webhook processing error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Internal server error',
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'gitlab-mr-webhook',
        'version': '2.0'
    })

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with service info."""
    return jsonify({
        'service': 'GitLab MR Webhook Service (Native)',
        'version': '2.0',
        'endpoints': {
            'webhook': '/webhook (POST)',
            'health': '/health (GET)'
        }
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    debug = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    
    logger.info(f"Starting GitLab MR Webhook Service (Native) on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
