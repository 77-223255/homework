"""
Watch Trained Frozen Lake Agent Play
====================================

See the trained agent navigate the frozen lake!

Author: Educational RL Project
"""

import numpy as np
import time
import os
from lake_env import FrozenLake, print_policy, print_value_function, print_policy_with_qmax
from agent import QLearningAgent


# =============================================================================
# CONFIGURATION
# =============================================================================

SAVE_PATH = os.path.join(os.path.dirname(__file__), 'trained_agent.pkl')
ANIMATION_DELAY = 0.5  # Seconds between moves


# =============================================================================
# PLAY FUNCTIONS
# =============================================================================

def watch_agent(agent, num_episodes=5, is_slippery=True, delay=ANIMATION_DELAY):
    """
    Watch the trained agent play.
    
    Args:
        agent: Trained Q-learning agent
        num_episodes: Number of episodes to show
        is_slippery: Ice slipperiness
        delay: Seconds between moves
    """
    env = FrozenLake(is_slippery=is_slippery)
    
    # Disable exploration
    old_epsilon = agent.exploration_rate
    agent.exploration_rate = 0.0
    
    print("\n" + "=" * 50)
    print("   🎬 WATCHING THE AGENT PLAY 🎬")
    print("=" * 50)
    print("\nLegend: A=Agent, H=Hole, G=Goal, S=Start, F=Frozen")
    print(f"Slippery ice: {is_slippery}")
    
    for episode in range(num_episodes):
        print(f"\n{'='*50}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print("=" * 50)
        
        state = env.reset()
        env.render()
        
        steps = 0
        done = False
        
        while not done:
            time.sleep(delay)
            
            # Get action
            action = agent.choose_action(state, training=False)
            action_name = env.ACTION_NAMES[action]
            
            print(f"Step {steps + 1}: {action_name}")
            
            # Take action
            next_state, reward, done = env.step(action)
            state = next_state
            steps += 1
            
            env.render()
            
            if done:
                if reward > 0:
                    print(f"🎉 SUCCESS! Reached goal in {steps} steps!")
                else:
                    print(f"💀 OOPS! Fell in a hole after {steps} steps!")
                break
        
        time.sleep(1)
    
    # Restore exploration
    agent.exploration_rate = old_epsilon


def interactive_play(agent, is_slippery=True):
    """
    Play alongside the agent (you choose actions).
    
    Args:
        agent: Trained agent (for suggestions)
        is_slippery: Ice slipperiness
    """
    env = FrozenLake(is_slippery=is_slippery)
    
    print("\n" + "=" * 50)
    print("   🎮 INTERACTIVE MODE 🎮")
    print("=" * 50)
    print("\nControls:")
    print("  0 = LEFT")
    print("  1 = DOWN")
    print("  2 = RIGHT")
    print("  3 = UP")
    print("  q = Quit")
    
    state = env.reset()
    env.render()
    
    steps = 0
    done = False
    
    while not done:
        # Show agent's suggestion
        agent_action = agent.choose_action(state, training=False)
        agent_suggestion = env.ACTION_NAMES[agent_action]
        print(f"💡 Agent suggests: {agent_suggestion}")
        
        # Get user input
        user_input = input("\nYour move (0-3, or q to quit): ").strip().lower()
        
        if user_input == 'q':
            print("Game quit.")
            return
        
        try:
            action = int(user_input)
            if action not in [0, 1, 2, 3]:
                print("Invalid action! Use 0, 1, 2, or 3.")
                continue
        except ValueError:
            print("Invalid input! Use a number 0-3.")
            continue
        
        # Take action
        state, reward, done = env.step(action)
        steps += 1
        
        print(f"\nYou chose: {env.ACTION_NAMES[action]}")
        env.render()
        
        if done:
            if reward > 0:
                print(f"🎉 You reached the goal in {steps} steps!")
            else:
                print(f"💀 You fell in a hole after {steps} steps!")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main function."""
    
    print("\n" + "=" * 60)
    print("   FROZEN LAKE: Watch Trained Agent")
    print("=" * 60)
    
    # Load trained agent
    agent = QLearningAgent()
    
    if not os.path.exists(SAVE_PATH):
        print("\n⚠️ No trained agent found!")
        print("Please run 'python train.py' first.")
        return
    
    agent.load(SAVE_PATH)
    agent.print_stats()
    
    # Show learned policy
    print_policy(agent)
    
    # Menu
    while True:
        print("\nWhat would you like to do?")
        print("  1. Watch agent play")
        print("  2. Play yourself (with agent suggestions)")
        print("  3. Show Q-max direction for each state")
        print("  4. Show value function")
        print("  5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '1':
            watch_agent(agent)
        elif choice == '2':
            interactive_play(agent)
        elif choice == '3':
            print_policy_with_qmax(agent)
        elif choice == '4':
            print_value_function(agent)
        elif choice == '5':
            print("\n👋 Bye!")
            break
        else:
            print("Invalid choice. Use 1-5.")


if __name__ == '__main__':
    main()