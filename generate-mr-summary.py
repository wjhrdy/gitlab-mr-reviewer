#!/usr/bin/env python3
"""
GitLab MR Diff Summary Generator using Gemini API
Automatically updates existing comments instead of creating new ones.
Supports self-signed certificates with VERIFY_SSL=false
Now supports external prompt templates for better maintainability.
"""

import os
import sys
import requests
import json
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional
from datetime import datetime
import google.generativeai as genai
import urllib3


class GitLabMRSummarizer:
    def __init__(self):
        self.gitlab_token = os.getenv('CI_JOB_TOKEN')
        self.project_id = os.getenv('CI_PROJECT_ID')
        self.mr_iid = os.getenv('CI_MERGE_REQUEST_IID')
        # Ensure GitLab URL is always HTTPS for security
        self.gitlab_url = os.getenv('CI_SERVER_URL', 'https://gitlab.cee.redhat.com/')
        if self.gitlab_url.startswith('http://'):
            self.gitlab_url = self.gitlab_url.replace('http://', 'https://')
            print(f"⚠️  Converted insecure URL to HTTPS: {self.gitlab_url}")
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        
        # Prompt template configuration
        self.prompt_template_path = os.getenv('PROMPT_TEMPLATE_PATH', '/app/prompt_template.txt')
        self.prompt_template_url = os.getenv('PROMPT_TEMPLATE_URL')  # Optional: load from URL
        
        if not all([self.gitlab_token, self.project_id, self.mr_iid, self.gemini_api_key]):
            missing = []
            if not self.gitlab_token: missing.append('CI_JOB_TOKEN')
            if not self.project_id: missing.append('CI_PROJECT_ID')
            if not self.mr_iid: missing.append('CI_MERGE_REQUEST_IID')
            if not self.gemini_api_key: missing.append('GEMINI_API_KEY')
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
        
        # Configure Gemini
        genai.configure(api_key=self.gemini_api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # GitLab API headers with secure configuration
        self.headers = {
            'Authorization': f'Bearer {self.gitlab_token}',
            'Content-Type': 'application/json',
            'User-Agent': 'GitLab-MR-Summary-Bot/1.0'
        }
        
        # Configure requests session for self-signed certificates
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # Disable SSL verification for self-signed certificates
        # Set VERIFY_SSL=false in CI/CD variables to disable SSL verification
        verify_ssl = os.getenv('VERIFY_SSL', 'false').lower() == 'true'
        self.session.verify = verify_ssl
        
        if not verify_ssl:
            print("⚠️  SSL verification disabled - allowing self-signed certificates")
            # Suppress SSL warnings when verification is disabled
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    def load_prompt_template(self) -> str:
        """Load prompt template from file, URL, or fallback to default."""
        
        # Try loading from URL first (if provided)
        if self.prompt_template_url:
            try:
                print(f"🌐 Loading prompt template from URL: {self.prompt_template_url}")
                response = requests.get(self.prompt_template_url, timeout=10)
                response.raise_for_status()
                return response.text
            except Exception as e:
                print(f"⚠️  Failed to load prompt from URL: {e}")
                print("Falling back to local file...")
        
        # Try loading from local file
        try:
            if os.path.exists(self.prompt_template_path):
                print(f"📄 Loading prompt template from file: {self.prompt_template_path}")
                with open(self.prompt_template_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                print(f"⚠️  Prompt template file not found: {self.prompt_template_path}")
        except Exception as e:
            print(f"⚠️  Failed to load prompt from file: {e}")
        
        # Fallback to embedded default prompt
        print("📝 Using embedded default prompt template")
        return self.get_default_prompt_template()
    
    def get_default_prompt_template(self) -> str:
        """Return the default embedded prompt template as fallback."""
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
        # Ensure URL is HTTPS but allow self-signed certs
        if url.startswith('http://'):
            url = url.replace('http://', 'https://')
            print(f"⚠️  Converted HTTP to HTTPS: {url.split('/')[2]}")
        
        try:
            # Use session's verify setting (respects VERIFY_SSL env var)
            response = self.session.request(method, url, timeout=30, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.SSLError as e:
            if self.session.verify:
                print(f"🔒 SSL Error - try setting VERIFY_SSL=false for self-signed certificates: {e}")
            else:
                print(f"🔒 SSL Error even with verification disabled: {e}")
            raise
        except requests.exceptions.Timeout:
            print("⏰ Request timeout - GitLab API may be slow")
            raise
        except requests.exceptions.RequestException as e:
            print(f"🌐 Network error: {e}")
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
            print(f"❌ Error fetching MR diff: {e}")
            return ""
    
    def get_mr_details(self) -> Dict:
        """Fetch merge request details."""
        url = f"{self.gitlab_url}/api/v4/projects/{self.project_id}/merge_requests/{self.mr_iid}"
        
        try:
            response = self.secure_api_request('GET', url)
            return response.json()
        except Exception as e:
            print(f"❌ Error fetching MR details: {e}")
            return {}
    
    def generate_summary_with_gemini(self, diff_content: str, mr_details: Dict) -> str:
        """Generate a summary using Gemini API with external prompt template."""
        
        # Truncate diff if it's too long (Gemini has token limits)
        max_diff_length = int(os.getenv('MAX_DIFF_LENGTH', '50000'))
        if len(diff_content) > max_diff_length:
            diff_content = diff_content[:max_diff_length] + "\n\n[...diff truncated...]"
        
        # Get current pipeline and commit info for context
        pipeline_url = os.getenv('CI_PIPELINE_URL', '')
        commit_sha = os.getenv('CI_COMMIT_SHA', '')[:8] if os.getenv('CI_COMMIT_SHA') else 'unknown'
        
        # Load the prompt template
        try:
            prompt_template = self.load_prompt_template()
        except Exception as e:
            print(f"❌ Error loading prompt template: {e}")
            prompt_template = self.get_default_prompt_template()
        
        # Format the prompt with actual values
        try:
            prompt = prompt_template.format(
                mr_title=mr_details.get('title', 'N/A'),
                mr_description=mr_details.get('description', 'N/A'),
                source_branch=mr_details.get('source_branch', 'N/A'),
                target_branch=mr_details.get('target_branch', 'N/A'),
                commit_sha=commit_sha,
                diff_content=diff_content,
                pipeline_url=pipeline_url
            )
        except KeyError as e:
            print(f"⚠️  Missing template variable {e}, using default template")
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
            
            # Look for our AI summary comment (identified by the unique header)
            comment_header = os.getenv('COMMENT_HEADER', '## 🤖 Gemini-Generated MR Summary')
            for note in notes:
                if note.get('body', '').startswith(comment_header):
                    return note['id']
            
            return None
        except Exception as e:
            print(f"❌ Error fetching MR comments: {e}")
            return None
    
    def post_or_update_mr_comment(self, summary: str) -> bool:
        """Post a new summary comment or update existing one."""
        pipeline_url = os.getenv('CI_PIPELINE_URL', '')
        # Ensure pipeline URL is HTTPS
        if pipeline_url.startswith('http://'):
            pipeline_url = pipeline_url.replace('http://', 'https://')
        
        commit_sha = os.getenv('CI_COMMIT_SHA', '')[:8] if os.getenv('CI_COMMIT_SHA') else 'unknown'
        comment_header = os.getenv('COMMENT_HEADER', '## 🤖 Gemini-Generated MR Summary')
        
        comment_body = f"""{comment_header}

{summary}

---
<details>
<summary>📋 Summary Details</summary>

- **Pipeline**: {f"[{commit_sha}]({pipeline_url})" if pipeline_url else commit_sha}
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
                print(f"✅ Successfully updated existing summary comment (ID: {existing_comment_id})")
                return True
            except Exception as e:
                print(f"❌ Error updating existing comment: {e}")
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
            print("✅ Successfully posted new summary comment to MR")
            return True
        except Exception as e:
            print(f"❌ Error posting new comment to MR: {e}")
            return False
    
    def get_current_timestamp(self) -> str:
        """Get current timestamp for comment updates."""
        from datetime import datetime
        return datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    
    def save_summary_artifacts(self, summary: str, mr_details: Dict):
        """Save summary as artifacts for the CI job."""
        
        # Save as markdown file
        with open('mr-summary.md', 'w') as f:
            f.write(f"# MR Summary: {mr_details.get('title', 'N/A')}\n\n")
            f.write(summary)
        
        # Create a simple JUnit XML report (for GitLab to pick up)
        root = ET.Element('testsuite')
        root.set('name', 'MR Summary Generation')
        root.set('tests', '1')
        root.set('failures', '0')
        root.set('errors', '0')
        
        testcase = ET.SubElement(root, 'testcase')
        testcase.set('name', 'generate_summary')
        testcase.set('classname', 'MRSummary')
        
        # Add summary as system-out
        system_out = ET.SubElement(testcase, 'system-out')
        system_out.text = summary
        
        tree = ET.ElementTree(root)
        tree.write('mr-summary-report.xml', encoding='utf-8', xml_declaration=True)
        
        print("Summary artifacts saved successfully")
    
    def run(self):
        """Main execution function."""
        print("Starting MR summary generation...")
        
        # Get MR details
        mr_details = self.get_mr_details()
        if not mr_details:
            print("Failed to fetch MR details")
            sys.exit(1)
        
        print(f"Processing MR: {mr_details.get('title', 'N/A')}")
        
        # Get diff
        diff_content = self.get_mr_diff()
        if not diff_content:
            print("No diff content found or error fetching diff")
            summary = "No changes detected in this merge request."
        else:
            print(f"Diff content length: {len(diff_content)} characters")
            
            # Generate summary
            print("Generating summary with Gemini...")
            summary = self.generate_summary_with_gemini(diff_content, mr_details)
        
        print("Generated summary:")
        print("-" * 50)
        print(summary)
        print("-" * 50)
        
        # Save artifacts
        self.save_summary_artifacts(summary, mr_details)
        
        # Post or update comment in MR
        if os.getenv('POST_COMMENT_TO_MR', 'true').lower() == 'true':
            self.post_or_update_mr_comment(summary)
        
        print("MR summary generation completed successfully!")


if __name__ == "__main__":
    try:
        summarizer = GitLabMRSummarizer()
        summarizer.run()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
