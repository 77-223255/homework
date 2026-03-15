"""
Watch Trained DQN Agent Play CartPole
=====================================

See the trained agent balance the pole!

Author: Educational RL Project
"""

import numpy as np
import os
import time

from cartpole_env import CartPole

try:
    from dqn_agent import DQNAgent, TORCH_AVAILABLE
except ImportError:
    TORCH_AVAILABLE = False

SAVE_PATH = os.path.join(os.path.dirname(__file__), 'dqn_model.pt')


def watch_agent(agent, num_episodes=5, delay=0.02):
    """
    Watch the trained agent play.
    
    Args:
        agent: Trained DQN agent
        num_episodes: Number of episodes to show
        delay: Delay between steps (seconds)
    """
    env = CartPole()
    
    # Disable exploration
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    print("\n" + "=" * 50)
    print("   🎬 WATCHING THE AGENT PLAY 🎬")
    print("=" * 50)
    
    scores = []
    
    for episode in range(num_episodes):
        print(f"\n{'='*50}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print("=" * 50)
        
        state = env.reset()
        score = 0
        step = 0
        
        print("\nInitial State:")
        env.render()
        
        for step in range(500):
            action = agent.choose_action(state, training=False)
            action_name = "RIGHT →" if action == 1 else "← LEFT"
            
            state, reward, done = env.step(action)
            score += reward
            
            if step % 20 == 0:
                print(f"\nStep {step:3d}: Action={action_name}")
                env.render()
            
            time.sleep(delay)
            
            if done:
                break
        
        scores.append(score)
        print(f"\n{'='*50}")
        print(f"Episode ended at step {step + 1}")
        print(f"Score: {score:.0f}")
        print("=" * 50)
    
    # Restore epsilon
    agent.epsilon = old_epsilon
    
    print(f"\n📊 Summary: Avg Score = {np.mean(scores):.1f}")


def compare_with_random(num_episodes=10):
    """
    Compare trained agent with random actions.
    
    Args:
        num_episodes: Number of episodes each
    """
    print("\n" + "=" * 60)
    print("   📊 COMPARISON: Trained vs Random")
    print("=" * 60)
    
    env = CartPole()
    
    # Random agent
    print(f"\nRandom Agent ({num_episodes} episodes)...")
    random_scores = []
    
    for _ in range(num_episodes):
        state = env.reset()
        score = 0
        
        for _ in range(500):
            action = np.random.randint(2)
            state, _, done = env.step(action)
            score += 1
            
            if done:
                break
        
        random_scores.append(score)
    
    print(f"  Average: {np.mean(random_scores):.1f}")
    print(f"  Max: {max(random_scores):.0f}")
    
    # Trained agent
    if TORCH_AVAILABLE and os.path.exists(SAVE_PATH):
        print(f"\nTrained Agent ({num_episodes} episodes)...")
        
        agent = DQNAgent()
        agent.load(SAVE_PATH)
        agent.epsilon = 0.0
        
        trained_scores = []
        
        for _ in range(num_episodes):
            state = env.reset()
            score = 0
            
            for _ in range(500):
                action = agent.choose_action(state, training=False)
                state, _, done = env.step(action)
                score += 1
                
                if done:
                    break
            
            trained_scores.append(score)
        
        print(f"  Average: {np.mean(trained_scores):.1f}")
        print(f"  Max: {max(trained_scores):.0f}")
        
        print(f"\n💡 Improvement: {np.mean(trained_scores) / np.mean(random_scores):.1f}x better!")
    else:
        print("\n⚠️ No trained model found. Run train.py first.")


def main():
    """Main function."""
    
    print("\n" + "=" * 60)
    print("   DQN CartPole: Watch Trained Agent")
    print("=" * 60)
    
    if not TORCH_AVAILABLE:
        print("\n❌ PyTorch is required!")
        print("Run: pip install torch")
        return
    
    # Load trained agent
    agent = DQNAgent()
    
    if not os.path.exists(SAVE_PATH):
        print("\n⚠️ No trained model found!")
        print("Please run 'python train.py' first.")
        return
    
    agent.load(SAVE_PATH)
    
    # Menu
    while True:
        print("\nWhat would you like to do?")
        print("  1. Watch agent play")
        print("  2. Compare with random")
        print("  3. Exit")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            watch_agent(agent)
        elif choice == '2':
            compare_with_random()
        elif choice == '3':
            print("\n👋 Bye!")
            break
        else:
            print("Invalid choice. Use 1-3.")


if __name__ == '__main__':
    main()