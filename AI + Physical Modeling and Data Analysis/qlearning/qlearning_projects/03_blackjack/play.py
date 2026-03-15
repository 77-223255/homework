"""
Play Blackjack Against Trained Agent
====================================

Watch the agent play or play yourself!

Author: Educational RL Project
"""

import os
from blackjack_env import Blackjack, state_to_string, print_strategy_table
from agent import BlackjackAgent, basic_strategy


# =============================================================================
# CONFIGURATION
# =============================================================================

SAVE_PATH = os.path.join(os.path.dirname(__file__), 'trained_agent.pkl')


# =============================================================================
# PLAY FUNCTIONS
# =============================================================================

def watch_agent(agent, num_games=5):
    """
    Watch the trained agent play.
    
    Args:
        agent: Trained agent
        num_games: Number of games to show
    """
    env = Blackjack()
    
    # Disable exploration
    old_epsilon = agent.exploration_rate
    agent.exploration_rate = 0.0
    
    print("\n" + "=" * 50)
    print("   🎬 WATCHING THE AGENT PLAY 🎬")
    print("=" * 50)
    
    wins = 0
    
    for game in range(num_games):
        print(f"\n{'='*50}")
        print(f"Game {game + 1}/{num_games}")
        print("=" * 50)
        
        state = env.reset()
        env.render()
        
        done = False
        actions_taken = []
        
        # Check for natural blackjack
        if state[0] == 21:
            dealer_sum = env._play_dealer()
            if dealer_sum == 21:
                reward = 0.0
            else:
                reward = 1.0
            done = True
            actions_taken = ["BLACKJACK!"]
        
        while not done:
            action = agent.choose_action(state, training=False)
            action_name = "HIT" if action == 1 else "STAND"
            actions_taken.append(action_name)
            
            print(f"Agent's decision: {action_name}")
            
            state, reward, done = env.step(action)
            env.render()
        
        # Show result
        player_sum = env._sum_hand(env.player_hand)
        dealer_sum = env._sum_hand(env.dealer_hand)
        
        print(f"\nFinal Result:")
        print(f"  Player: {player_sum}")
        print(f"  Dealer: {dealer_sum}")
        
        if reward > 0:
            print("  🎉 WIN!")
            wins += 1
        elif reward < 0:
            print("  💀 LOSE!")
        else:
            print("  🤝 DRAW!")
        
        print(f"  Actions: {' -> '.join(actions_taken)}")
    
    # Restore exploration
    agent.exploration_rate = old_epsilon
    
    print(f"\n{'='*50}")
    print(f"Summary: {wins}/{num_games} wins ({wins/num_games*100:.0f}%)")
    print("=" * 50)


def play_game(agent):
    """
    Play a game of Blackjack yourself.
    
    Args:
        agent: Trained agent (for suggestions)
    """
    env = Blackjack()
    
    print("\n" + "=" * 50)
    print("   🎮 BLACKJACK 🎮")
    print("=" * 50)
    print("\nRules:")
    print("  - Get closer to 21 than dealer without going over")
    print("  - Hit = draw another card")
    print("  - Stand = stop drawing")
    print("  - Going over 21 = BUST = lose!")
    
    state = env.reset()
    env.render()
    
    done = False
    
    # Check for natural blackjack
    if state[0] == 21:
        dealer_sum = env._play_dealer()
        if dealer_sum == 21:
            print("\n🎉 Both have Blackjack! DRAW!")
        else:
            print("\n🎉 BLACKJACK! YOU WIN!")
        return
    
    while not done:
        # Show agent's suggestion
        agent_action = agent.choose_action(state, training=False)
        agent_suggestion = "HIT" if agent_action == 1 else "STAND"
        basic_action = basic_strategy(state)
        basic_suggestion = "HIT" if basic_action == 1 else "STAND"
        
        print(f"\n💡 Agent suggests: {agent_suggestion}")
        print(f"   Basic strategy: {basic_suggestion}")
        
        # Get user input
        user_input = input("\nYour choice (h=hit, s=stand, q=quit): ").strip().lower()
        
        if user_input == 'q':
            print("Game quit.")
            return
        elif user_input == 'h':
            action = 1
        elif user_input == 's':
            action = 0
        else:
            print("Invalid input! Use 'h', 's', or 'q'.")
            continue
        
        state, reward, done = env.step(action)
        env.render()
        
        if done:
            player_sum = env._sum_hand(env.player_hand)
            dealer_sum = env._sum_hand(env.dealer_hand)
            
            print(f"\nFinal Result:")
            print(f"  Player: {player_sum}")
            print(f"  Dealer: {dealer_sum}")
            
            if reward > 0:
                print("  🎉 YOU WIN!")
            elif reward < 0:
                print("  💀 YOU LOSE!")
            else:
                print("  🤝 DRAW!")


def test_yourself(agent, num_games=20):
    """
    Test your Blackjack skills against the agent.
    
    Args:
        agent: Trained agent
        num_games: Number of games to test
    """
    print("\n" + "=" * 60)
    print("   🎯 BLACKJACK SKILL TEST 🎯")
    print("=" * 60)
    print(f"\nYou'll play {num_games} games.")
    print("The agent will tell you the correct play after each decision.")
    
    env = Blackjack()
    
    player_wins = 0
    correct_decisions = 0
    total_decisions = 0
    
    for game in range(num_games):
        print(f"\n{'='*40}")
        print(f"Game {game + 1}/{num_games}")
        print("=" * 40)
        
        state = env.reset()
        done = False
        
        # Check for natural blackjack
        if state[0] == 21:
            dealer_sum = env._play_dealer()
            if dealer_sum == 21:
                print("🤝 Both have Blackjack! Draw!")
            else:
                print("🎉 Blackjack! You win!")
                player_wins += 1
            continue
        
        while not done:
            player_sum, dealer_showing, usable_ace = state
            env.render()
            
            # Get optimal action
            optimal = basic_strategy(state)
            optimal_name = "HIT" if optimal == 1 else "STAND"
            
            # Get user choice
            user_input = input("Your choice (h=hit, s=stand): ").strip().lower()
            
            if user_input == 'h':
                action = 1
            elif user_input == 's':
                action = 0
            else:
                print("Invalid! Defaulting to STAND.")
                action = 0
            
            # Check if correct
            total_decisions += 1
            if action == optimal:
                correct_decisions += 1
                print("✅ Correct!")
            else:
                print(f"❌ Wrong! Optimal was: {optimal_name}")
            
            state, reward, done = env.step(action)
        
        if reward > 0:
            player_wins += 1
            print("🎉 You won this game!")
        elif reward < 0:
            print("💀 You lost this game!")
        else:
            print("🤝 Draw!")
    
    # Final stats
    print("\n" + "=" * 60)
    print("   📊 YOUR RESULTS")
    print("=" * 60)
    print(f"  Win rate: {player_wins}/{num_games} ({player_wins/num_games*100:.1f}%)")
    print(f"  Correct decisions: {correct_decisions}/{total_decisions} ({correct_decisions/total_decisions*100:.1f}%)")
    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main function."""
    
    print("\n" + "=" * 60)
    print("   BLACKJACK: Play Against Trained Agent")
    print("=" * 60)
    
    # Load trained agent
    agent = BlackjackAgent()
    
    if not os.path.exists(SAVE_PATH):
        print("\n⚠️ No trained agent found!")
        print("Please run 'python train.py' first.")
        return
    
    agent.load(SAVE_PATH)
    agent.print_stats()
    
    # Menu
    while True:
        print("\nWhat would you like to do?")
        print("  1. Watch agent play")
        print("  2. Play a game yourself")
        print("  3. Test your skills (with feedback)")
        print("  4. Show strategy table")
        print("  5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '1':
            watch_agent(agent)
        elif choice == '2':
            play_game(agent)
        elif choice == '3':
            test_yourself(agent)
        elif choice == '4':
            print_strategy_table(agent)
        elif choice == '5':
            print("\n👋 Thanks for playing!")
            break
        else:
            print("Invalid choice. Use 1-5.")


if __name__ == '__main__':
    main()