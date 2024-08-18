import random
import math
import matplotlib.pyplot as plt

#number of games played can be changed
NUM_GAMES = 50

#creates the board and handles the game logic (movement, who wins)
class tic_tac_toe:
    def __init__(self):
        #initialize a 3x3 board with empty spaces
        self.board = [' ' for _ in range(9)]
        self.current_winner = None  #track the winner

    def print_board(self):
        for row in [self.board[i*3:(i+1)*3] for i in range(3)]: #slices the baord into 3 rows
            print('| ' + ' | '.join(row) + ' |') #adds dividers for each elmnt
        print()

    def empty_squares(self):
        #check if there are any empty squares left, returns true/false
        return ' ' in self.board

    def available_moves(self):
        #return a list of indexes where the board is empty
        return [i for i, x in enumerate(self.board) if x == ' ']

    def make_move(self, square, letter):
        #place a letter on the board at the given square if it's empty
        if self.board[square] == ' ':
            self.board[square] = letter
            #check if this move wins the game
            if self.winner(square, letter):
                self.current_winner = letter
            return True
        return False

    def winner(self, square, letter):
        #check for a winning combination at that row
        row_ind = square // 3  #row index
        row = self.board[row_ind*3:(row_ind+1)*3]  #row content
        if all([s == letter for s in row]):
            return True
        #checks the combo at the column
        col_ind = square % 3  #column index
        column = [self.board[col_ind+i*3] for i in range(3)]  #column content
        if all([s == letter for s in column]):
            return True
        #check the combo at diagonals
        if square % 2 == 0: #even numbered squares have diagonals
            diagonal1 = [self.board[i] for i in [0, 4, 8]]  #top left to bottom right
            if all([s == letter for s in diagonal1]):
                return True
            diagonal2 = [self.board[i] for i in [2, 4, 6]] #top right to bottom left
            if all([s == letter for s in diagonal2]):
                return True
        return False

#minimax with alphabeta pruning agent
class minimax:
    def __init__(self, letter):
        self.letter = letter
        self.node_count = 0

    def get_action(self, game):
        self.node_count = 0
        best_score = -math.inf if self.letter == 'X' else math.inf
        best_move = None

        for move in game.available_moves(): #loop through available moves
            game.make_move(move, self.letter)
            #recursively calculate score
            score = self.minimax(game, -math.inf, math.inf, False) if self.letter == 'X' else self.minimax(game, -math.inf, math.inf, True)
            game.board[move] = ' '  #resets the move
            game.current_winner = None
            #update score based on minimax
            if self.letter == 'X' and score > best_score:
                best_score = score
                best_move = move
            elif self.letter == 'O' and score < best_score:
                best_score = score
                best_move = move

        return best_move, self.node_count

    def minimax(self, game, alpha, beta, is_maximizing):
        self.node_count += 1
        #check for any winners
        if game.current_winner == 'X':
            return 1
        elif game.current_winner == 'O':
            return -1
        elif game.empty_squares() == False:
            return 0

        if is_maximizing:
            best_score = -math.inf
            for move in game.available_moves(): #explore possible moves
                game.make_move(move, 'X')
                score = self.minimax(game, alpha, beta, False)  #recursive
                game.board[move] = ' '  #redo
                game.current_winner = None  #redo
                best_score = max(score, best_score)
                alpha = max(alpha, score)
                if beta <= alpha:
                    break   #prune the remaining branches
            return best_score
        else:
            best_score = math.inf
            for move in game.available_moves(): #explore possible moves
                game.make_move(move, 'O')
                score = self.minimax(game, alpha, beta, True) #recursive
                game.board[move] = ' '   #redo
                game.current_winner = None  #redo
                best_score = min(score, best_score)
                beta = min(beta, score)
                if beta <= alpha:
                    break   #prune
            return best_score

class mcts_node:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.wins = 0

    def fully_explored(self):    #if all possible nodes have been explored
        return len(self.children) == len(self.state.available_moves())

    def best_child(self, c_param=1.4):  #returns child with the highest ucb
        best_value = -float('inf')
        best_node = None
        for child in self.children:
            exploitation = child.wins / child.visits
            exploration = c_param * math.sqrt(2 * math.log(self.visits) / child.visits)
            ucb1_value = exploitation + exploration
            if ucb1_value > best_value:
                best_value = ucb1_value
                best_node = child
        return best_node

#mcts algorithm agent
class monte_carlo:
    def __init__(self, letter, simulations=50):    #num of simulations can be changes
        self.letter = letter
        self.simulations = simulations
        self.node_count = 0

    def get_action(self, game):
        self.node_count = 0
        root = mcts_node(game)
        
        for _ in range(self.simulations):
            node = self.select(root)
            reward = self.simulate(node)
            self.backpropagate(node, reward)
        #best child with most visits
        best_move = max(root.children, key=lambda child: child.visits).move
        return best_move, self.node_count

    def select(self, node): #select a ndoe to expand
        while not node.state.current_winner and node.state.empty_squares():
            self.node_count += 1
            if not node.fully_explored():
                return self.expand(node)
            else:   #select best node
                node = node.best_child()
        return node

    def expand(self, node): #expand node that haven't been explored
        untried_moves = [move for move in node.state.available_moves() if move not in [child.move for child in node.children]]
        move = random.choice(untried_moves)
        next_state = tic_tac_toe()
        next_state.board = node.state.board[:]
        next_state.make_move(move, self.letter if node.state.board.count('X') == node.state.board.count('O') else ('O' if self.letter == 'X' else 'X'))
        child_node = mcts_node(next_state, parent=node, move=move)
        node.children.append(child_node)
        return child_node

    def simulate(self, node):   #simulates the game for that node
        temp_game = tic_tac_toe()
        temp_game.board = node.state.board[:]
        temp_game.current_winner = node.state.current_winner
        current_letter = 'O' if self.letter == 'X' else 'X'
        #simulate a game until it ends
        while temp_game.empty_squares() and not temp_game.current_winner:
            move = self.select_move(temp_game, current_letter)
            temp_game.make_move(move, current_letter)
            current_letter = 'O' if current_letter == 'X' else 'X'
        
        if temp_game.current_winner == self.letter:
            return 1
        elif temp_game.current_winner is None:
            return 0
        else:
            return -1

    def select_move(self, game, letter):
        for move in game.available_moves(): #check if a move can win the game
            temp_game = tic_tac_toe()
            temp_game.board = game.board[:]
            temp_game.make_move(move, letter)
            if temp_game.winner(move, letter):
                return move
        
        opponent_letter = 'O' if letter == 'X' else 'X' #check if it can block opponent
        for move in game.available_moves():
            temp_game = tic_tac_toe()
            temp_game.board = game.board[:]
            temp_game.make_move(move, opponent_letter)
            if temp_game.winner(move, opponent_letter):
                return move
        
        return random.choice(game.available_moves()) #else, choose random move

    def backpropagate(self, node, reward): #update node as we come back
        while node is not None:
            node.visits += 1    #increment visits to the node
            node.wins += reward
            node = node.parent
            reward = -reward    #reverse for opp perspective

#plays the tic-tac-toe game, looping through number of games playing
def play_game(search_algo):
    x_wins = 0
    o_wins = 0
    draws = 0
    minimax_node = 0
    montecarlo_node = 0
    nodes_visted = 0
    total_rounds = 0
    x_wins_list = []
    o_wins_list = []
    monte_visited_list = []
    mini_visited_list = []

    #play the number of games wanted
    for game_num in range(NUM_GAMES):
        print(f"Game {game_num + 1}:")
        game = tic_tac_toe()
        if search_algo == 1:
            x_agent = monte_carlo('X')  #X is the Monte Carlo agent
        else:
            x_agent = minimax('X')  #X is the Minimax agent
        rounds = 0

        while game.empty_squares():
            #O makes a random move
            o_move = random.choice(game.available_moves())
            game.make_move(o_move, 'O')
            rounds += 1
            game.print_board()

            if game.current_winner:
                print(f"{game.current_winner} wins!\n")
                o_wins += 1
                break
            elif game.empty_squares() == False:
                print("It's a draw!\n")
                draws += 1
                break

            #X makes a move using one of the algos
            move, nodes_visted = x_agent.get_action(game)
            if search_algo == 1:
                montecarlo_node += nodes_visted
            else :
                minimax_node += nodes_visted
            game.make_move(move, 'X')
            rounds += 1
            game.print_board()

            if game.current_winner:
                print(f"{game.current_winner} wins!\n")
                x_wins += 1
                break
            elif game.empty_squares() == False:
                print("It's a draw!\n")
                draws += 1
                break

        #add up averages and scores for the game
        total_rounds += rounds
        x_wins_list.append(x_wins)  #append to the list for plot
        o_wins_list.append(o_wins)
        if search_algo == 1:    #append to the list for plot
            monte_visited_list.append(nodes_visted)
        else:
            mini_visited_list.append(nodes_visted)

    avg_rounds = total_rounds / NUM_GAMES   #average rounds played per game
    if search_algo == 1:
        return x_wins, o_wins, draws, avg_rounds, x_wins_list, o_wins_list, monte_visited_list
    return x_wins, o_wins, draws, avg_rounds, x_wins_list, o_wins_list, mini_visited_list

def main():    
    monte_x_wins, monte_o_wins, monte_draws, monte_avg_rounds, monte_x_wins_list, monte_o_wins_list, monte_nodes_visited = play_game(1)
    mini_x_wins, mini_o_wins, mini_draws, mini_avg_rounds, mini_x_wins_list, mini_o_wins_list, minimax_nodes_visited = play_game(0)
    monte_avg_node = 0
    mini_avg_node = 0
    for i in monte_nodes_visited:   #get average
        monte_avg_node += i
    for i in minimax_nodes_visited: #get average
        mini_avg_node += i
    monte_avg_node /= NUM_GAMES
    mini_avg_node /= NUM_GAMES
    monte_win_rate = (monte_x_wins / (monte_x_wins + monte_o_wins + monte_draws)) * 100
    mini_win_rate = (mini_x_wins / (mini_o_wins + mini_x_wins + mini_draws)) * 100
    print("Monte Carlo")
    print(f"X wins: {monte_x_wins}, O wins: {monte_o_wins}, Draws: {monte_draws}, Win rate: {monte_win_rate}%, Average rounds per game: {monte_avg_rounds:.2f}, Average nodes visited per game: {monte_avg_node}")
    print("Minimax")
    print(f"X wins: {mini_x_wins}, O wins: {mini_o_wins}, Draws: {mini_draws}, Win rate: {mini_win_rate}%, Average rounds per game: {mini_avg_rounds:.2f}, Average nodes visited per game: {mini_avg_node}")
    
    plt.figure(figsize=(12, 6))
    #plot monte carlo tree search results
    plt.plot(range(1, NUM_GAMES + 1), monte_x_wins_list, marker='x', linestyle='--', color='green', label='Monte Carlo X Wins')
    plt.plot(range(1, NUM_GAMES + 1), monte_o_wins_list, marker='x', linestyle='--', color='blue', label='Monte Carlo O Wins')
    #plot minimax alphabeta pruning results
    plt.plot(range(1, NUM_GAMES + 1), mini_x_wins_list, marker='o', linestyle='-', color='orange', label='Minimax X Wins')
    plt.plot(range(1, NUM_GAMES + 1), mini_o_wins_list, marker='o', linestyle='-', color='red', label='Minimax O Wins')
    plt.xlabel('Number of Games')
    plt.ylabel('Cumulative Wins')
    plt.title('Minimax vs. Monte Carlo: Cumulative Wins for X and O')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    #plot nodes visited
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, NUM_GAMES + 1), monte_nodes_visited, marker='x', linestyle='--', color='purple', label='Monte Carlo Nodes Visited')
    plt.plot(range(1, NUM_GAMES + 1), minimax_nodes_visited, marker='o', linestyle='-', color='black', label='Minimax Nodes Visited')
    plt.xlabel('Number of Games')
    plt.ylabel('Nodes Visited')
    plt.title('Nodes Visited: Monte Carlo vs Minimax')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()