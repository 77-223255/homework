"""
Blackjack Environment
=====================

A simplified Blackjack game for Q-learning.

Rules:
- Goal: Get closer to 21 than dealer without going over
- Hit: Draw another card
- Stand: Stop drawing
- Bust: Go over 21 = lose
- Dealer hits until 17+
- Ace counts as 1 or 11 (whichever is better)

State: (player_sum, dealer_showing, usable_ace)
- player_sum: 12-21 (below 12 is always hit)
- dealer_showing: 1-10 (Ace = 1)
- usable_ace: True if player has Ace counting as 11

Author: Educational RL Project
"""

import numpy as np
import random


# =============================================================================
# BLACKJACK ENVIRONMENT
# =============================================================================

class Blackjack:
    """
    A simplified Blackjack environment.
    
    The agent plays against a dealer with fixed strategy (hit until 17).
    
    Attributes:
        player_hand: List of player's cards
        dealer_hand: List of dealer's cards
        dealer_showing: Dealer's visible card
    """
    
    # Actions
    STAND = 0
    HIT = 1
    
    ACTION_NAMES = ['STAND', 'HIT']
    
    def __init__(self):
        """Initialize the Blackjack environment."""
        self.player_hand = []
        self.dealer_hand = []
        self.dealer_showing = None
        self.reset()
    
    def reset(self):
        """
        Start a new game.
        
        Returns:
            state: (player_sum, dealer_showing, usable_ace)
        """
        self.player_hand = []
        self.dealer_hand = []
        
        # Deal initial cards
        # Player gets 2 cards
        self.player_hand.append(self._draw_card())
        self.player_hand.append(self._draw_card())
        
        # Dealer gets 2 cards (one face up, one face down)
        self.dealer_hand.append(self._draw_card())  # Face up
        self.dealer_hand.append(self._draw_card())  # Face down
        
        self.dealer_showing = self.dealer_hand[0]
        
        return self._get_state()
    
    def _draw_card(self):
        """
        Draw a card from infinite deck.
        
        Returns:
            Card value (1-10, where 1 = Ace)
        """
        # Infinite deck: probability based on 52-card deck
        # Cards 1-9: 4 suits each = 4/52 = 1/13
        # Cards 10, J, Q, K: 4 suits × 4 cards = 16/52 = 4/13
        probs = [1/13] * 9 + [4/13]  # [1,2,3,4,5,6,7,8,9,10]
        return np.random.choice(range(1, 11), p=probs)
    
    def _sum_hand(self, hand):
        """
        Calculate the sum of a hand.
        
        Aces count as 11 if it doesn't cause bust.
        
        Args:
            hand: List of card values
            
        Returns:
            Sum of hand
        """
        total = sum(hand)
        aces = hand.count(1)
        
        # Convert Aces from 1 to 11 if beneficial
        for _ in range(aces):
            if total + 10 <= 21:
                total += 10
        
        return total
    
    def _has_usable_ace(self, hand):
        """
        Check if hand has a usable Ace (counting as 11).
        
        Args:
            hand: List of card values
            
        Returns:
            True if Ace counts as 11
        """
        total = sum(hand)
        aces = hand.count(1)
        
        for _ in range(aces):
            if total + 10 <= 21:
                return True
        
        return False
    
    def _get_state(self):
        """
        Get current state.
        
        Returns:
            tuple: (player_sum, dealer_showing, usable_ace)
        """
        player_sum = self._sum_hand(self.player_hand)
        usable_ace = self._has_usable_ace(self.player_hand)
        
        return (player_sum, self.dealer_showing, usable_ace)
    
    def step(self, action):
        """
        Take an action.
        
        Args:
            action: 0 = STAND, 1 = HIT
            
        Returns:
            next_state: New state
            reward: Reward received
            done: Whether game ended
        """
        if action == self.HIT:
            # Draw a card
            self.player_hand.append(self._draw_card())
            player_sum = self._sum_hand(self.player_hand)
            
            if player_sum > 21:
                # Bust! Player loses
                return self._get_state(), -1.0, True
            elif player_sum == 21:
                # Perfect! Dealer's turn
                return self._stand()
            else:
                # Continue playing
                return self._get_state(), 0.0, False
        
        else:  # STAND
            return self._stand()
    
    def _stand(self):
        """
        Player stands. Dealer plays.
        
        Returns:
            state: Final state
            reward: Game result
            done: Always True
        """
        player_sum = self._sum_hand(self.player_hand)
        dealer_sum = self._play_dealer()
        
        # Determine winner
        if dealer_sum > 21:
            # Dealer busts, player wins
            return self._get_state(), 1.0, True
        elif player_sum > dealer_sum:
            # Player wins
            return self._get_state(), 1.0, True
        elif player_sum < dealer_sum:
            # Dealer wins
            return self._get_state(), -1.0, True
        else:
            # Draw
            return self._get_state(), 0.0, True
    
    def _play_dealer(self):
        """
        Dealer plays according to fixed strategy (hit until 17).
        
        Returns:
            Dealer's final sum
        """
        while self._sum_hand(self.dealer_hand) < 17:
            self.dealer_hand.append(self._draw_card())
        
        return self._sum_hand(self.dealer_hand)
    
    def render(self):
        """Print current game state."""
        player_sum = self._sum_hand(self.player_hand)
        dealer_sum = self._sum_hand(self.dealer_hand[:1])  # Only showing card
        
        print("\n" + "=" * 40)
        print("  🃏 BLACKJACK 🃏")
        print("=" * 40)
        print(f"  Dealer showing: {self.dealer_showing} (hidden card: ?)")
        print(f"  Your hand: {self.player_hand} = {player_sum}")
        
        if self._has_usable_ace(self.player_hand):
            print("  (You have a usable Ace!)")
        print("=" * 40)
    
    def get_state_space(self):
        """
        Get all possible states.
        
        Returns:
            List of all possible states
        """
        states = []
        
        # Player sum: 4-21 (below 4 would keep hitting)
        # Actually, we track 12-21 (below 12 always hit)
        for player_sum in range(12, 22):
            for dealer_showing in range(1, 11):  # 1-10
                for usable_ace in [False, True]:
                    states.append((player_sum, dealer_showing, usable_ace))
        
        return states


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def state_to_string(state):
    """
    Convert state to readable string.
    
    Args:
        state: (player_sum, dealer_showing, usable_ace)
        
    Returns:
        String representation
    """
    player_sum, dealer_showing, usable_ace = state
    ace_str = "A" if usable_ace else " "
    return f"(You:{player_sum:2d}, Dealer:{dealer_showing:2d}, Ace:{ace_str})"


def print_strategy_table(agent):
    """
    Print the learned strategy in a table format.
    
    Args:
        agent: Trained Q-learning agent
    """
    print("\n" + "=" * 80)
    print("   📊 LEARNED BLACKJACK STRATEGY")
    print("=" * 80)
    print("\nLegend: S = STAND, H = HIT")
    print("\nPlayer Sum | Dealer Showing (1-10, 1=Ace)")
    print("-" * 60)
    
    # Header
    header = "           | "
    for d in range(1, 11):
        d_str = "A" if d == 1 else str(d)
        header += f"{d_str:>4} "
    print(header)
    print("-" * 60)
    
    # No usable ace
    print("No Usable Ace:")
    for player_sum in range(20, 11, -1):
        row = f"    {player_sum:2d}     | "
        for dealer in range(1, 11):
            state = (player_sum, dealer, False)
            if state in agent.q_table:
                q = agent.q_table[state]
                best = 'S' if q[0] >= q[1] else 'H'
                row += f"  {best}  "
            else:
                row += "  ?  "
        print(row)
    
    print()
    
    # With usable ace
    print("With Usable Ace:")
    for player_sum in range(20, 11, -1):
        row = f"    {player_sum:2d}     | "
        for dealer in range(1, 11):
            state = (player_sum, dealer, True)
            if state in agent.q_table:
                q = agent.q_table[state]
                best = 'S' if q[0] >= q[1] else 'H'
                row += f"  {best}  "
            else:
                row += "  ?  "
        print(row)
    
    print("=" * 80)
    print("\n💡 Compare with 'Basic Strategy' charts online!")


# =============================================================================
# TEST
# =============================================================================

def test_environment():
    """Test the Blackjack environment."""
    print("=" * 50)
    print("   BLACKJACK ENVIRONMENT TEST")
    print("=" * 50)
    
    env = Blackjack()
    
    for game in range(3):
        print(f"\n--- Game {game + 1} ---")
        state = env.reset()
        env.render()
        
        done = False
        while not done:
            # Simple strategy: hit if below 17
            player_sum = state[0]
            if player_sum < 17:
                action = 1  # HIT
                print("Decision: HIT (below 17)")
            else:
                action = 0  # STAND
                print("Decision: STAND (17+)")
            
            state, reward, done = env.step(action)
            env.render()
        
        if reward > 0:
            print("🎉 You WIN!")
        elif reward < 0:
            print("💀 You LOSE!")
        else:
            print("🤝 DRAW!")


if __name__ == '__main__':
    test_environment()