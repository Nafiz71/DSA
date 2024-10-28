#Name: Nafiz Ibraham
#Student ID: 32699247
#Assignment 2

#============================================== Task 1 ==============================================

from collections import deque
# Class to represent the flow network
class FlowNetwork:
    def __init__(self, n):
        """
        Initialize the flow network with an empty adjacency list and capacity matrix.
        Time Complexity: O(n^2)
        Space Complexity: O(n^2)

        """
        self.graph = [[] for _ in range(n)]  # Adjacency list -- O(n)
        self.capacity = [[0] * n for _ in range(n)]  # O(n^2)
        self.n = n  # Number of nodes ~ no of participants 

    def add_edge(self, u, v, capacity):
        """
        Adds an edge to the flow network and assigns capacity.
        The graph is directed from u to v with the given capacity.

        Time & Space Complexity: O(1)
        Return: None

        """
        #FC: O(1)
        self.graph[u].append(v)
        self.graph[v].append(u)  # Add reverse edge for residual graph
        self.capacity[u][v] = capacity  # Assign capacity to the forward edge

    def bfs(self, source, sink, parent):
        """
        BFS to find an augmenting path in the residual graph. It returns True if there
        is a path from source to sink, and also fills the parent array to store the path.

        Time Complexity: O(V+E)
        Space Complexity: O(V)
        Return: Boolean

        """
        visited = [False] * self.n  # Keep track of visited nodes --O(n)
        queue = deque([source])  # Initialize queue with the source 
        visited[source] = True  # Mark the source as visited
        
        while queue: #O(V)
            u = queue.popleft()  # Pop a node from the queue --O(1)
            
            for v in self.graph[u]:  # Traverse neighbors of node u --O(E)
                if not visited[v] and self.capacity[u][v] > 0:  # Unvisited and has available capacity
                    parent[v] = u  # Store the path to v via u
                    visited[v] = True  # Mark v as visited
                    if v == sink:  # If we reach the sink, we found a path
                        return True
                    queue.append(v)  # Otherwise, keep exploring by adding v to the queue
        return False  # No path found

    def ford_fulkerson(self, source, sink):
        """
        Implementation of the Ford-Fulkerson/ Edmonds Karp method using BFS to find the maximum flow from
        the source to the sink in the flow network.

        Note since using BFS to find augmenting path , it's Edmonds Karp but just used Ford Fulkerson term
        as its actually a variant of Ford Fulkerson

        V = n + 2 * m + 2
        E = O(n * m)
        Therefore, the time complexity becomes O((n + 2 m + 2) (n m)^2).
        Simplified to O(|𝑉||𝐸|2)

        Time Complexity: O((|𝑉||𝐸|2) 
        Auxiliary Space: O(n)
        """
        #O(VE^2)
        parent = [-1] * self.n  # To store the augmenting path
        max_flow = 0  # Initialize max flow to zero
        
        # While there is an augmenting path from source to sink
        while self.bfs(source, sink, parent):
            # Find the maximum flow through the path found by BFS
            path_flow = float('inf')
            v = sink
            while v != source:
                u = parent[v]
                path_flow = min(path_flow, self.capacity[u][v])  # Bottleneck capacity
                v = u
            
            # Update residual capacities of the edges and reverse edges along the path
            v = sink
            while v != source:
                u = parent[v]
                self.capacity[u][v] -= path_flow  # Decrease capacity along the forward path
                self.capacity[v][u] += path_flow  # Increase capacity along the reverse path
                v = u
            
            # Add path flow to the overall flow
            max_flow += path_flow
        
        return max_flow

def assign(preferences, places):
    """
    My Approach/Strategy: 

    For visual understanding  of my approach refer to the flow network diagram I've created below :) note this is not the actual flow network but just to understand my approach 
    Network Structure:

                    Leaders (cap=2)
                   ┌──── L0(A0) ───┐
          ┌─[P0]───┤               │
          │        │               │
          ├─[P1]───┤               │
[SOURCE]──┤        │               ├──[SINK]
          ├─[P2]───┤    L1(A1)     │
          │        │               │
          └─[P3]───┘               │
                    └── M0,M1 ─────┘
                     Members (cap=places-2)

                -> Applying Ford Fulkerson  or Edmonds Karp algorithm to find max flow 

    My approach is to build a flow network to stracture the participanmts and activites as nodes 
    - We connect the participants to the source and we connect the activity to sink 
    - So each activity is connected to the sink node ; we created one set for leaders and one set for non leaders 
    where there capacity is 2 for l nodes and non leaders number of places/spots reserved - 2 i.e leaders have 2 places reserved and non leaders have places-2 reserved 
    -We then connect the participants to activity so atleast one partscipants has one activity and at least leader for each activity 
    - Then we perform fulkerson to mind the max flow 
    - Since all p needs to be allocated to an activity , hence the max flow needs to be same as n participants , if max flow is equal to number of participants
    - it's a valid assignment :)


    #this is my rough analysis a
    Rough= #FF flownetwork + source to participants + connecting to participants +activities to sink + ford fulkerson + max flow check 
    - Building the network: O(n * m), where n is the number of participants and m is the number of activities.
    - Ford-Fulkerson maximum flow: O(n^3), as we perform the max flow algorithm on a graph with O(n) nodes.
    - Overall: O(n^3)


    Complexity Analysis:
    - Initialization and setup: O(1)
    - Flow network initialization: O(n^2) / 
    - Connect source to participants: O(n)
    - Connect participants to activities: O(n^2)/ Nested loop over n participants and m activities: O(n m)
    - Connect activities to sink: O(m)
    - Ford-Fulkerson algorithm: 𝑂(|𝑉||𝐸|2) ~O(n^3)
    - Constructing assignments: O(n^2)

    Overall Complexity: O(n^3), where n is the number of participants.
    Space Complexity: O(n^2)
    Auxiliary Space: O(n)
    
    
    Parameters:
    - preferences: List of lists indicating each participant's preferences for activities.
    - places: List indicating the number of places available for each activity.

    Returns:
    - A list of assignments if a valid assignment exists, otherwise None.
    """
    #F T.C = #FF flownetwork + source to participants + connecting to participants +activities to sink + ford fulkerson + max flow check 
    n = len(preferences)  # Number of participants
    m = len(places)  # Number of activities

    # Total number of nodes in the flow network
    source = n + 2 * m  # Source node index
    sink = source + 1  # Sink node index
    total_nodes = n + 2 * m + 2  # Total number of nodes including source and sink

    #Initializing the flow network
    flow_network = FlowNetwork(total_nodes) #O(n)

    # Connect source to participants
    for i in range(n):
        flow_network.add_edge(source, i, 1)  # Source to each participant with capacity 1 -O(n)

    # Connect participants to activities based on their preferences
    #O(n^2)
    # O(n*m), m<-(n/2), (n*n/2)- O(n^2)
    for i in range(n): #O(n)
        for j in range(m): #O(m)
            if preferences[i][j] == 2:  # If the person has experience (leader candidate)
                flow_network.add_edge(i, n + j, 1)  # Connect to the leader node for activity j
            if preferences[i][j] > 0:  # If interested in the activity
                flow_network.add_edge(i, n + m + j, 1)  # Connect to the member node for activity j

    # Connect activities to sink (for leaders and members)
    for j in range(m): #O(n/2) ~ O(n)
        flow_network.add_edge(n + j, sink, 2)  # Leader nodes (capacity 2) ~ O(1)
        flow_network.add_edge(n + m + j, sink, places[j] - 2)  # Member nodes (remaining places) refer to the diagram for more clarity

    # Run the Ford-Fulkerson / Edmonds Karp algorithm to find maximum flow
    #O(VE^2)
    max_flow = flow_network.ford_fulkerson(source, sink) # O𝑂(|𝑉||𝐸|2)~(O(n^3)

    # If I max_flow, it equals the number of participants, we have a valid assignment
    # Proof of correctness as our week 1 study guide and achieve decent  marks :))
    if max_flow == n:
        assignments = [[] for _ in range(m)]
        for i in range(n):
            for j in range(m):
                if flow_network.capacity[n + j][i] > 0:  # Participant i is assigned to activity j as leader
                    assignments[j].append(i)
                elif flow_network.capacity[n + m + j][i] > 0:  # Participant i is assigned to activity j as a regular member
                    assignments[j].append(i)
        return assignments
    else:
        return None

# Example usage:
# preferences = [[2, 1], [2, 1], [2, 0], [1, 2], [1, 2]]
# places = [2, 3]
# print(assign(preferences,places))
# print(assign([[2, 0, 0, 0, 1, 1, 0, 2, 2, 1], [0, 0, 2, 1, 0, 1, 1, 1, 1, 0], [0, 1, 2, 0, 2, 1, 2, 1, 1, 1], [2, 2, 1, 1, 1, 1, 1, 2, 0, 0], [0, 1, 0, 0, 0, 1, 0, 2, 0, 0], [1, 0, 1, 1, 1, 0, 0, 2, 2, 0], [2, 2, 0, 0, 2, 1, 0, 1, 0, 1], [2, 2, 2, 1, 1, 2, 0, 0, 2, 1], [0, 2, 0, 1, 0, 0, 0, 0, 0, 2], [2, 1, 0, 2, 1, 0, 1, 0, 1, 0], [1, 1, 1, 2, 1, 2, 1, 2, 0, 0], [0, 1, 1, 2, 2, 1, 2, 1, 2, 1], [0, 2, 1, 0, 1, 2, 1, 2, 2, 2], [1, 2, 2, 1, 2, 0, 0, 1, 0, 2], [1, 0, 0, 1, 1, 1, 0, 0, 1, 2], [1, 2, 2, 2, 0, 2, 0, 1, 0, 0], [0, 0, 1, 1, 1, 2, 0, 0, 2, 1], [1, 0, 0, 1, 0, 0, 2, 2, 1, 1], [1, 0, 1, 1, 0, 0, 1, 0, 2, 2], [0, 1, 0, 1, 2, 0, 2, 1, 0, 0], [0, 0, 2, 0, 0, 0, 2, 2, 0, 1], [1, 1, 2, 0, 1, 1, 2, 0, 1, 1], [1, 0, 0, 2, 1, 1, 2, 0, 0, 1], [2, 0, 0, 2, 0, 0, 1, 1, 0, 2], [2, 0, 0, 1, 1, 2, 0, 1, 2, 2], [0, 1, 0, 2, 0, 0, 1, 1, 0, 0], [2, 2, 0, 1, 2, 2, 2, 1, 0, 2], [1, 1, 2, 0, 0, 1, 1, 1, 0, 0], [0, 1, 2, 1, 0, 2, 1, 1, 2, 2], [0, 1, 0, 1, 1, 1, 1, 0, 1, 1], [1, 2, 2, 1, 0, 2, 1, 0, 2, 1], [2, 1, 0, 1, 2, 0, 0, 1, 2, 1], [2, 0, 1, 2, 2, 2, 1, 1, 0, 0], [2, 1, 1, 0, 0, 0, 1, 2, 1, 1], [1, 1, 2, 2, 0, 0, 1, 1, 0, 2], [2, 1, 0, 1, 2, 2, 2, 0, 2, 0], [0, 0, 2, 1, 2, 0, 0, 2, 2, 1], [1, 0, 0, 0, 0, 0, 2, 1, 1, 1], [0, 2, 1, 0, 2, 2, 0, 1, 2, 1], [0, 1, 0, 1, 0, 0, 2, 1, 0, 0], [2, 0, 2, 0, 1, 0, 0, 1, 0, 0], [1, 0, 1, 0, 0, 0, 1, 0, 2, 1], [1, 2, 0, 2, 1, 0, 2, 1, 0, 1], [0, 0, 0, 0, 2, 1, 0, 1, 2, 0], [2, 2, 1, 2, 0, 2, 2, 0, 2, 1], [2, 1, 0, 1, 1, 0, 1, 2, 0, 1], [2, 2, 1, 1, 1, 1, 2, 1, 1, 2], [2, 2, 1, 0, 0, 1, 0, 1, 2, 2], [0, 1, 2, 2, 1, 1, 1, 2, 1, 2], [0, 1, 1, 2, 0, 0, 0, 1, 1, 1], [2, 0, 2, 0, 2, 2, 1, 1, 0, 0], [1, 1, 0, 1, 0, 0, 2, 1, 1, 0], [1, 0, 2, 2, 2, 1, 1, 2, 1, 1], [0, 0, 1, 0, 0, 0, 1, 0, 0, 2], [1, 1, 1, 0, 2, 0, 2, 0, 2, 2], [2, 0, 1, 2, 1, 2, 2, 2, 1, 1]], [9, 5, 8, 4, 9, 6, 2, 7, 3, 3]))
# print(assign(preferences, places))  # Expected output: [[0, 3], [1, 4, 2]]




#============================================== Task 2 ==============================================

#Nafiz Ibraham
#32699247


#Nafiz Ibraham
#32699247
#============================================== Task 2 ==============================================

class TrieNode:

    """
    TrieNode class represents a node in the Trie data structure needed for Trie initialization.
    """
    def __init__(self):
        # Initialize node with links, end flag, frequency, and word for the trie
        # Time & Space: O(1)
        self.link = [None] * 63  # 63 characters includes all uppercase, lowercase, and digits 
        self.is_end = False  # end of word
        self.frequency = 0
        self.word = None

    def get_index(self, char: str) -> int:
        # Convert character to index in link array
        # Time: O(1), Space: O(1)
        # Return the index of the character in the link array or -1 if the character is not found
        if char == '$': 
            return 0
        elif 'a' <= char <= 'z':
            return ord(char) - ord('a') + 1
        elif 'A' <= char <= 'Z':
            return ord(char) - ord('A') + 27
        elif '0' <= char <= '9':
            return ord(char) - ord('0') + 53
        return -1


class Trie:
    
    """
    Trie class represents the Trie data structure, which initializes the root node and builds the Trie based on the input file and by inserting words into the Trie.
    """
    def __init__(self):
        # Initialize Trie with root node
        # Time & Space: O(1)
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        """
        Insert a word into the Trie.
        First insert the (through characters) word into the Trie by traversing the word and creating new nodes if necessary.
        Then, if the word already exists in the Trie, increase the frequency of the word.
        If the word doesn't exist, set the is_end flag to True, frequency to 1, and word to the input word.

        Time: O(M), Space: O(M) where M is word length
        Return nothing / Just inserting the word into the trie
        """
        current = self.root
        for char in word:
            index = current.get_index(char)
            if index == -1:
                continue
            if current.link[index] is None:
                current.link[index] = TrieNode()
            current = current.link[index]
        if current.is_end:  # we increase the frequency of the word if it already exists
            current.frequency += 1
        else:
            current.is_end = True
            current.frequency = 1
            current.word = word

    def search(self, word: str) -> bool:
        """
        A bit of same implementation as insert but here we just return true if the word exists.
        Search for word in Trie
        Time: O(M), Space: O(1) where M is word length
        Returns: True if the word exists
        """
        current = self.root
        for char in word:
            index = current.get_index(char)
            if index == -1 or current.link[index] is None:
                return False
            current = current.link[index]
        return current.is_end  # return true if the word exists

    def gatherwords(self, node, max_words, words_collected):
        """
        Collect words from Trie node; just collecting the words that exist not the final suggestion
        Time: O(K), Space: O(K) where K is number of collected words
        Returns: True if the number of collected words is greater than or equal to max_words
        """
        if node.is_end: # means that the word exists
            words_collected.append((node.word, node.frequency))
            if len(words_collected) >= max_words:
                return True
        # Recursively collect words from all children nodes
        for child in node.link:
            if child is not None:
                if self.gatherwords(child, max_words, words_collected):
                    return True
        return False

    @staticmethod
    def samelen_prefix(s1: str, s2: str) -> int:
        # Calculate common prefix length of two strings
        # Time: O(min(len(s1), len(s2))), Space: O(1)
        # Return the length of the common prefix
        length = min(len(s1), len(s2))
        for i in range(length): # O(min(len(s1), len(s2)))~ O(M)
            if s1[i] != s2[i]:
                return i
        return length


class SpellChecker:
    def __init__(self, filename: str) -> None:
        """
        Initialize SpellChecker with Trie from file
        Time: O(T), Space: O(T) where T is total characters in file (Messages.txt)

        """
        self.trie = Trie()
        self.build_trie(filename) # O(T)

    def build_trie(self, filename: str) -> None:
        # Build Trie from file; here it's Messages.txt
        # Time: O(T), Space: O(T) where T is total characters in file
        with open(filename, 'r') as file:
            word = []
            for char in file.read():
                if char.isalnum():
                    word.append(char)
                elif word:
                    self.trie.insert(''.join(word))
                    word = []
            if word:
                self.trie.insert(''.join(word))

    def check(self, input_word: str) -> list:
        # Check spelling and suggest corrections
        # Time: O(M + U), Space: O(1)
        # Where M is input word length, U is number of Trie nodes
        """
      
        This function checks the spelling of an input word and provides suggestions if it's misspelled
        (Using the three priority rules, the most same prefix length, the frequency of the word, and the lexicographical order).

        My Approach description:
        To solve this task, I implemented a trie(prefix tree)-based spell checker:
        1. First, I check if the word exists in my trie. If it does, I return an empty list since it's spelled correctly.
        2. If the word isn't found, I do a forward scan of the input word:
           - At each prefix point, I collect words from my trie that share the current prefix.
           - For each word I collect, I calculate how much of a prefix it shares with the input word.
           - I keep a list of the best suggestions, ordering them by prefix length, frequency, and alphabetically.
        3. Finally, I return the top 3 suggestions (or fewer if there aren't 3).

        I chose this approach because tries are really efficient for prefix-based operations. By scanning 
        forward and collecting words at each point, I make sure I don't miss any potential matches, even 
        if they diverge early from the input word. The tricky part was balancing efficiency with accuracy - 
        I had to be careful not to explore too much of the trie at each step, which is why I limit how many 
        words I collect each time.

        break down the time complexity:
        T(M, U) = O(M) + M * (O(1) + O(1) + O(M))
     
       However, this analysis assumes worst-case behavior for same_prefix_length and collect_words. In practice:
        1. same_prefix_length often terminates early, not always taking O(M) time.
        2. collect_words explores a portion of the trie, which we can represent as O(U), where U is the number of nodes in the trie.
        3. Therefore,we can say a more accurate representation of the time complexity is:

        T(M, U) = O(M) + O(U)


        Input:
        input_word: A string representing the word to be spell-checked.

        Output/ return:
        A list of strings containing up to 3 spelling suggestions. If the input word is correctly spelled
        (exists in the trie), an empty list is returned.

        Time complexity:
        after deducing  check above ^
        Best case: O(M)
        Worst case: O(M + U)
        Average case/Big O: O(M + U)
        Where M is the length of the input word and U is the number of nodes in the trie.

        Time complexity analysis:
      
        Best case: When the word exists in the trie, I only traverse the input word once: O(M).
        Worst case: I traverse the input word (O(M)) and potentially explore the entire trie at each
        prefix point (O(U)). Collecting and processing suggestions at each point is bounded by
        a constant factor, so it doesn't affect the overall complexity.
        Things to remember 

        Space and auxiliary space complexity:
        O(1)

        Space and auxiliary space complexity analysis:
        only using a constant amount of extra space for the suggestions list (max 3 items) and
        a few variables. The space I use doesn't grow with the input size or trie size.
        """
        # Early exit for existing words
        if self.trie.search(input_word):
            return []  # best case  O(M)

        suggestions = []
        max_suggestions = 3
        current = self.trie.root
        prefix_length = 0

        # Forward scan to collect words at each prefix point
        for char in input_word:  # O(M)
            index = current.get_index(char)
            if index == -1 or current.link[index] is None:
                break
            current = current.link[index]
            prefix_length += 1

            # Collect words at current prefix point
            words_from_current_prefix = []
            self.trie.gatherwords(current, max_suggestions * 2, words_from_current_prefix)
            
            # Process collected words
            for word, freq in words_from_current_prefix:
                common_len = self.trie.samelen_prefix(input_word, word)
                if common_len > 0:
                    self._insert_suggestion(suggestions, (common_len, freq, word))

        # Return top suggestions
        return [item[2] for item in suggestions[:max_suggestions]]  # O(1) since fixed size list

    def _insert_suggestion(self, suggestions, new_suggestion):
        """
        Insert a new suggestion into our list of top suggestions.

        I'm maintaining a tiny leaderboard of the best word suggestions.
        I scan through current top suggestions, updating or inserting the new one
        based on our ranking rules. This keeps our top 3 always sorted.

        Time: O(1) due to fixed size list, Space: O(1)
        Return nothing
        """
        for i, existing in enumerate(suggestions):  # O(1) since assuming/ theoritically fixed size list
            if new_suggestion[2] == existing[2]:
                if new_suggestion[1] > existing[1]:
                    suggestions[i] = new_suggestion
                return
            if self._compare_suggestions(new_suggestion, existing):
                suggestions.insert(i, new_suggestion)
                return
        if len(suggestions) < 3:
            suggestions.append(new_suggestion)

    def _compare_suggestions(self, new, existing):
        """Compare suggestions maintaining correct ordering v.v imp
            Time: O(1), Space: O(1)
            Return true if new is greater than existing
        """
        # First by common prefix length
        if new[0] != existing[0]:
            return new[0] > existing[0]
        # Then by frequency
        if new[1] != existing[1]:
            return new[1] > existing[1]
        # Finally by lexicographical order as per rule of the task priority 
        new_word, existing_word = new[2], existing[2]
        max_len = max(len(new_word), len(existing_word))
        new_word = new_word.ljust(max_len, '\0')
        existing_word = existing_word.ljust(max_len, '\0')
        return new_word < existing_word


# myChecker = SpellChecker("/Users/admin/Monash/units/FIT2004/a2/A2work/Messages.txt")
# print(myChecker.check("IDK"))   # Expected Output：[]
# print(myChecker.check("zoo"))   # Expected Output：[]
# print(myChecker.check("LOK"))   # Expected Output：["LOL", "LMK"]
# print(myChecker.check("IDP"))   # Expected Output：["IDK", "IDC", "I"]
# print(myChecker.check("Ifc"))   # Expected Output：["If", "I", "IDK"]

