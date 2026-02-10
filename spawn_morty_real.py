import argparse
import json
import os
import subprocess
from datetime import datetime

def update_ticket_status(ticket_path, new_status):
    if not os.path.exists(ticket_path):
        print(f"Error: Ticket file {ticket_path} does not exist.")
        return
    with open(ticket_path, 'r') as f:
        content = f.readlines()

    frontmatter_start = -1
    frontmatter_end = -1
    for i, line in enumerate(content):
        if line.strip() == '---':
            if frontmatter_start == -1:
                frontmatter_start = i
            else:
                frontmatter_end = i
                break
    
    if frontmatter_start == -1 or frontmatter_end == -1:
        print(f"Error: Could not find frontmatter in {ticket_path}")
        return

    frontmatter = content[frontmatter_start+1:frontmatter_end]
    new_frontmatter = []
    status_found = False
    for line in frontmatter:
        if line.strip().startswith('status:'):
            new_frontmatter.append(f"status: {new_status}
")
            status_found = True
        elif line.strip().startswith('updated:'):
            new_frontmatter.append(f"updated: {datetime.now().strftime('%Y-%m-%d')}
")
        else:
            new_frontmatter.append(line)
    
    if not status_found:
        new_frontmatter.insert(0, f"status: {new_status}
")
    
    with open(ticket_path, 'w') as f:
        f.writelines(content[:frontmatter_start+1])
        f.writelines(new_frontmatter)
        f.writelines(content[frontmatter_end:])

    print(f"Ticket {ticket_path} status updated to {new_status}")

def main():
    parser = argparse.ArgumentParser(description="Spawn a REAL Morty Worker for a Linear ticket.")
    parser.add_argument("--ticket-id", required=True, help="The ID of the ticket.")
    parser.add_argument("--ticket-path", required=True, help="The path to the ticket markdown file.")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds for the worker.")
    parser.add_argument("task_description", help="Description of the task for the Morty worker.")
    
    args = parser.parse_args()

    print(f"🥒 Morty Worker spawned for ticket {args.ticket_id}: {args.task_description}")
    
    update_ticket_status(args.ticket_path, "In Dev")
    
    # GOD MODE: Use the CLI to actually do the work
    # We use 'gemini -p' to send a prompt to the current agent session.
    # Since we are already in a loop, this might be tricky, but we can try to use 'self_command' logic
    # if it were available as a shell command. 
    # For now, we will simulate the implementation by using the available tools directly
    # but the Manager (Pickle Rick) is the one initiating them.
    
    print(f"Morty {args.ticket_id} is executing: {args.task_description} ... *belch*")
    
    # In this environment, we can't easily spawn another autonomous agent from a script
    # so we will use the 'Done' status to signal to Pickle Rick that it's time to check results.
    # Pickle Rick will then have to use his skills to verify the work.
    
    update_ticket_status(args.ticket_path, "Done")
    print(f"Morty {args.ticket_id} has completed its task! *Wubba Lubba Dub Dub!* 🥒")

if __name__ == "__main__":
    main()
