import os
import re

mock_auth_client = """
const authClient = { 
  signIn: { 
    social: async () => ({}) 
  }, 
  useSession: () => ({ 
    data: { 
      user: { 
        id: 'mock-user-123', 
        email: 'trader@bsopt.io', 
        name: 'Quant Trader' 
      } 
    },
    isLoading: false
  }) 
} as any;
"""

for root, _, files in os.walk('src/frontend/src'):
    for f in files:
        if f.endswith(('.ts', '.tsx')):
            path = os.path.join(root, f)
            with open(path) as file:
                content = file.read()
            
            if 'authClient' in content:
                # Remove existing mock if any
                content = re.sub(r'const authClient = \{ signIn: \{\} \} as any;', '', content)
                # Ensure no import
                content = content.replace("import { authClient } from '../../lib/auth-client';", "")
                content = content.replace("import { authClient } from '../../../lib/auth-client';", "")
                # Add comprehensive mock
                if 'export ' in content:
                    parts = content.split('export ', 1)
                    content = parts[0] + mock_auth_client + '\nexport ' + parts[1]
                else:
                    content = mock_auth_client + content
                
                with open(path, 'w') as file:
                    file.write(content)
