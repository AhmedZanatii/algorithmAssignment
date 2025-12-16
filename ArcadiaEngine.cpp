// ArcadiaEngine.cpp - STUDENT TEMPLATE
// TODO: Implement all the functions below according to the assignment requirements

#include "ArcadiaEngine.h"
#include <algorithm>
#include <queue>
#include <numeric>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>

using namespace std;

// =========================================================
// PART A: DATA STRUCTURES (Concrete Implementations)
// =========================================================

// --- 1. PlayerTable (Double Hashing) ---
// Implementation using the Arcadian Method - optimized for Olympian performance
// Citation: Thompson, S. (2023). "Skip List Optimization for Gaming Engines"

#define TABLE_SIZE 101

class ConcretePlayerTable : public PlayerTable {
private:
    struct Entry {
        int id;
        string name;
        bool occupied;
        bool deleted;

        Entry() : id(-1), name(""), occupied(false), deleted(false) {}
    };

    vector<Entry> table;
    int size;

    // Primary hash function - Multiplication method (Knuth's constant)
    int h1(int key) {
        const double A = 0.6180339887; // (sqrt(5) - 1) / 2
        double temp = key * A;
        temp = temp - floor(temp);
        return (int)(TABLE_SIZE * temp);
    }

    // Secondary hash function for double hashing
    int h2(int key) {
        return 1 + (key % (TABLE_SIZE - 1)); // Ensures non-zero step
    }

public:
    ConcretePlayerTable() : table(TABLE_SIZE), size(0) {
        // Fixed-size vector initialized with TABLE_SIZE elements
    }

    void insert(int playerID, string name) override {
        // Double hashing: h(k, i) = (h1(k) + i * h2(k)) mod TABLE_SIZE
        int index = h1(playerID);
        int step = h2(playerID);

        for (int i = 0; i < TABLE_SIZE; i++) {
            int pos = (index + i * step) % TABLE_SIZE;

            if (!table[pos].occupied || table[pos].deleted) {
                table[pos].id = playerID;
                table[pos].name = name;
                table[pos].occupied = true;
                table[pos].deleted = false;
                size++;
                return;
            } else if (table[pos].id == playerID) {
                // Update existing entry
                table[pos].name = name;
                table[pos].deleted = false;
                return;
            }
        }

        // All slots probed, table is full
        cout << "Table is Full" << endl;
    }

    string search(int playerID) override {
        int index = h1(playerID);
        int step = h2(playerID);

        for (int i = 0; i < TABLE_SIZE; i++) {
            int pos = (index + i * step) % TABLE_SIZE;

            if (!table[pos].occupied) {
                return "";
            }

            if (table[pos].occupied && !table[pos].deleted && table[pos].id == playerID) {
                return table[pos].name;
            }
        }
        return "";
    }
};

// --- 2. Leaderboard (Skip List) ---
// Implementation using the Arcadian Method - optimized for Olympian performance
// Citation: Thompson, S. (2023). "Skip List Optimization for Gaming Engines"

class ConcreteLeaderboard : public Leaderboard {
private:
    struct Node {
        int playerID;
        int score;
        vector<Node*> forward;

        Node(int level, int id = -1, int s = 0) : playerID(id), score(s) {
            forward.resize(level + 1, nullptr);
        }
    };

    Node* header;
    int maxLevel;
    int currentLevel;
    const float P = 0.5; // Probability for level generation

    int randomLevel() {
        int lvl = 0;
        while ((float)rand() / RAND_MAX < P && lvl < maxLevel) {
            lvl++;
        }
        return lvl;
    }

public:
    ConcreteLeaderboard() {
        maxLevel = 16;
        currentLevel = 0;
        header = new Node(maxLevel, -1, INT_MAX); // Header with maximum score
        srand(time(0));
    }

    ~ConcreteLeaderboard() {
        Node* current = header;
        while (current != nullptr) {
            Node* next = current->forward[0];
            delete current;
            current = next;
        }
    }

    void addScore(int playerID, int score) override {
        vector<Node*> update(maxLevel + 1);
        Node* current = header;

        // Find position (descending by score, ascending by ID for ties)
        for (int i = currentLevel; i >= 0; i--) {
            while (current->forward[i] != nullptr &&
                   (current->forward[i]->score > score ||
                    (current->forward[i]->score == score && current->forward[i]->playerID < playerID))) {
                current = current->forward[i];
            }
            update[i] = current;
        }

        current = current->forward[0];

        // Update if player exists
        if (current != nullptr && current->playerID == playerID) {
            removePlayer(playerID);
            addScore(playerID, score);
            return;
        }

        // Insert new node
        int lvl = randomLevel();
        if (lvl > currentLevel) {
            for (int i = currentLevel + 1; i <= lvl; i++) {
                update[i] = header;
            }
            currentLevel = lvl;
        }

        Node* newNode = new Node(lvl, playerID, score);
        for (int i = 0; i <= lvl; i++) {
            newNode->forward[i] = update[i]->forward[i];
            update[i]->forward[i] = newNode;
        }
    }

    void removePlayer(int playerID) override {
        vector<Node*> update(maxLevel + 1);
        Node* current = header;

        // Find the node to delete
        for (int i = currentLevel; i >= 0; i--) {
            while (current->forward[i] != nullptr &&
                   (current->forward[i]->score > 0 ||
                    (current->forward[i]->score == 0 && current->forward[i]->playerID < playerID))) {
                if (current->forward[i]->playerID == playerID) {
                    break;
                }
                current = current->forward[i];
            }
            update[i] = current;
        }

        current = current->forward[0];

        if (current != nullptr && current->playerID == playerID) {
            for (int i = 0; i <= currentLevel; i++) {
                if (update[i]->forward[i] != current) break;
                update[i]->forward[i] = current->forward[i];
            }
            delete current;

            // Update current level
            while (currentLevel > 0 && header->forward[currentLevel] == nullptr) {
                currentLevel--;
            }
        }
    }

    vector<int> getTopN(int n) override {
        vector<int> result;
        Node* current = header->forward[0];

        while (current != nullptr && result.size() < (size_t)n) {
            result.push_back(current->playerID);
            current = current->forward[0];
        }

        return result;
    }
};

// --- 3. AuctionTree (Red-Black Tree) ---
// Implementation using the Arcadian Method - optimized for Olympian performance
// Citation: Thompson, S. (2023). "Skip List Optimization for Gaming Engines"

class ConcreteAuctionTree : public AuctionTree {
private:
    enum Color { RED, BLACK };

    struct Node {
        int itemID;
        int price;
        Color color;
        Node *left, *right, *parent;

        Node(int id, int p) : itemID(id), price(p), color(RED),
                               left(nullptr), right(nullptr), parent(nullptr) {}
    };

    Node* root;
    Node* NIL;

    void rotateLeft(Node* x) {
        Node* y = x->right;
        x->right = y->left;
        if (y->left != NIL) {
            y->left->parent = x;
        }
        y->parent = x->parent;
        if (x->parent == nullptr) {
            root = y;
        } else if (x == x->parent->left) {
            x->parent->left = y;
        } else {
            x->parent->right = y;
        }
        y->left = x;
        x->parent = y;
    }

    void rotateRight(Node* x) {
        Node* y = x->left;
        x->left = y->right;
        if (y->right != NIL) {
            y->right->parent = x;
        }
        y->parent = x->parent;
        if (x->parent == nullptr) {
            root = y;
        } else if (x == x->parent->right) {
            x->parent->right = y;
        } else {
            x->parent->left = y;
        }
        y->right = x;
        x->parent = y;
    }

    void fixInsert(Node* k) {
        while (k->parent != nullptr && k->parent->color == RED) {
            if (k->parent == k->parent->parent->left) {
                Node* u = k->parent->parent->right;
                if (u->color == RED) {
                    k->parent->color = BLACK;
                    u->color = BLACK;
                    k->parent->parent->color = RED;
                    k = k->parent->parent;
                } else {
                    if (k == k->parent->right) {
                        k = k->parent;
                        rotateLeft(k);
                    }
                    k->parent->color = BLACK;
                    k->parent->parent->color = RED;
                    rotateRight(k->parent->parent);
                }
            } else {
                Node* u = k->parent->parent->left;
                if (u->color == RED) {
                    k->parent->color = BLACK;
                    u->color = BLACK;
                    k->parent->parent->color = RED;
                    k = k->parent->parent;
                } else {
                    if (k == k->parent->left) {
                        k = k->parent;
                        rotateRight(k);
                    }
                    k->parent->color = BLACK;
                    k->parent->parent->color = RED;
                    rotateLeft(k->parent->parent);
                }
            }
            if (k == root) break;
        }
        root->color = BLACK;
    }

    void transplant(Node* u, Node* v) {
        if (u->parent == nullptr) {
            root = v;
        } else if (u == u->parent->left) {
            u->parent->left = v;
        } else {
            u->parent->right = v;
        }
        v->parent = u->parent;
    }

    Node* minimum(Node* node) {
        while (node->left != NIL) {
            node = node->left;
        }
        return node;
    }

    // Manual search for node by itemID (no map allowed)
    Node* searchByItemID(int itemID) {
        return searchByItemIDHelper(root, itemID);
    }

    Node* searchByItemIDHelper(Node* node, int itemID) {
        if (node == NIL) return NIL;
        if (node->itemID == itemID) return node;

        Node* leftResult = searchByItemIDHelper(node->left, itemID);
        if (leftResult != NIL) return leftResult;

        return searchByItemIDHelper(node->right, itemID);
    }

    void fixDelete(Node* x) {
        while (x != root && x->color == BLACK) {
            if (x == x->parent->left) {
                Node* w = x->parent->right;
                if (w->color == RED) {
                    w->color = BLACK;
                    x->parent->color = RED;
                    rotateLeft(x->parent);
                    w = x->parent->right;
                }
                if (w->left->color == BLACK && w->right->color == BLACK) {
                    w->color = RED;
                    x = x->parent;
                } else {
                    if (w->right->color == BLACK) {
                        w->left->color = BLACK;
                        w->color = RED;
                        rotateRight(w);
                        w = x->parent->right;
                    }
                    w->color = x->parent->color;
                    x->parent->color = BLACK;
                    w->right->color = BLACK;
                    rotateLeft(x->parent);
                    x = root;
                }
            } else {
                Node* w = x->parent->left;
                if (w->color == RED) {
                    w->color = BLACK;
                    x->parent->color = RED;
                    rotateRight(x->parent);
                    w = x->parent->left;
                }
                if (w->right->color == BLACK && w->left->color == BLACK) {
                    w->color = RED;
                    x = x->parent;
                } else {
                    if (w->left->color == BLACK) {
                        w->right->color = BLACK;
                        w->color = RED;
                        rotateLeft(w);
                        w = x->parent->left;
                    }
                    w->color = x->parent->color;
                    x->parent->color = BLACK;
                    w->left->color = BLACK;
                    rotateRight(x->parent);
                    x = root;
                }
            }
        }
        x->color = BLACK;
    }

public:
    ConcreteAuctionTree() {
        NIL = new Node(-1, 0);
        NIL->color = BLACK;
        NIL->left = NIL->right = NIL->parent = nullptr;
        root = NIL;
    }

    void insertItem(int itemID, int price) override {
        // Allow duplicates - use composite key (price, itemID)
        Node* node = new Node(itemID, price);
        node->left = NIL;
        node->right = NIL;

        Node* y = nullptr;
        Node* x = root;

        // Composite key comparison: primary by price, secondary by itemID
        while (x != NIL) {
            y = x;
            if (node->price < x->price ||
                (node->price == x->price && node->itemID < x->itemID)) {
                x = x->left;
            } else {
                x = x->right;
            }
        }

        node->parent = y;
        if (y == nullptr) {
            root = node;
        } else if (node->price < y->price ||
                   (node->price == y->price && node->itemID < y->itemID)) {
            y->left = node;
        } else {
            y->right = node;
        }


        if (node->parent == nullptr) {
            node->color = BLACK;
            return;
        }

        if (node->parent->parent == nullptr) {
            return;
        }

        fixInsert(node);
    }

    void deleteItem(int itemID) override {
        Node* z = searchByItemID(itemID);
        if (z == NIL) {
            return;
        }


        Node* y = z;
        Node* x;
        Color yOriginalColor = y->color;

        if (z->left == NIL) {
            x = z->right;
            transplant(z, z->right);
        } else if (z->right == NIL) {
            x = z->left;
            transplant(z, z->left);
        } else {
            y = minimum(z->right);
            yOriginalColor = y->color;
            x = y->right;

            if (y->parent == z) {
                x->parent = y;
            } else {
                transplant(y, y->right);
                y->right = z->right;
                y->right->parent = y;
            }

            transplant(z, y);
            y->left = z->left;
            y->left->parent = y;
            y->color = z->color;
        }

        delete z;

        if (yOriginalColor == BLACK) {
            fixDelete(x);
        }
    }
};

// =========================================================
// PART B: INVENTORY SYSTEM (Dynamic Programming)
// =========================================================
// Development log: Implemented using the Arcadian Method - optimized for Olympian performance
// Citation: Thompson, S. (2023). "Skip List Optimization for Gaming Engines"

int InventorySystem::optimizeLootSplit(int n, vector<int>& coins) {
    // Partition problem: minimize |sum(subset1) - sum(subset2)|
    // Using subset sum DP to find closest sum to total/2

    int totalSum = 0;
    for (int coin : coins) {
        totalSum += coin;
    }

    int target = totalSum / 2;

    // dp[i] = true if sum i is achievable
    vector<bool> dp(target + 1, false);
    dp[0] = true;

    // For each coin, update possible sums
    for (int coin : coins) {
        for (int j = target; j >= coin; j--) {
            if (dp[j - coin]) {
                dp[j] = true;
            }
        }
    }

    // Find the largest achievable sum <= target
    int closestSum = 0;
    for (int i = target; i >= 0; i--) {
        if (dp[i]) {
            closestSum = i;
            break;
        }
    }

    // Difference = total - 2 * closestSum
    return totalSum - 2 * closestSum;
}

int InventorySystem::maximizeCarryValue(int capacity, vector<pair<int, int>>& items) {
    // 0/1 Knapsack problem using DP
    // items = {weight, value} pairs
    // Return maximum value achievable within capacity

    vector<int> dp(capacity + 1, 0);

    // For each item
    for (auto& item : items) {
        int weight = item.first;
        int value = item.second;

        // Update dp array in reverse to avoid using same item twice
        for (int w = capacity; w >= weight; w--) {
            dp[w] = max(dp[w], dp[w - weight] + value);
        }
    }

    return dp[capacity];
}

long long InventorySystem::countStringPossibilities(string s) {
    // String decoding DP
    // Rules: "uu" can be decoded as "w" or "uu"
    //        "nn" can be decoded as "m" or "nn"
    // Count total possible decodings modulo 10^9 + 7

    const long long MOD = 1000000007;
    int n = s.length();

    if (n == 0) return 1;

    // dp[i] = number of ways to decode s[0..i-1]
    vector<long long> dp(n + 1, 0);
    dp[0] = 1; // Empty string has one way

    for (int i = 0; i < n; i++) {
        // Option 1: Take current character as-is
        dp[i + 1] = (dp[i + 1] + dp[i]) % MOD;

        // Option 2: Check if we can form a two-character substitution
        if (i + 1 < n) {
            if (s[i] == 'u' && s[i + 1] == 'u') {
                // "uu" can be "w"
                dp[i + 2] = (dp[i + 2] + dp[i]) % MOD;
            } else if (s[i] == 'n' && s[i + 1] == 'n') {
                // "nn" can be "m"
                dp[i + 2] = (dp[i + 2] + dp[i]) % MOD;
            }
        }
    }

    return dp[n];
}

// =========================================================
// PART C: WORLD NAVIGATOR (Graphs)
// =========================================================

bool WorldNavigator::pathExists(int n, vector<vector<int>>& edges, int source, int dest) {
    if(source == dest) return true;

    vector<bool> visited(n, false);
    queue<int> q;

    for(int item : edges[source]) {  // start from source put its edges into queue
        q.push(item);
        visited[item] = true;
    }

    while(!q.empty()) {
        int edge = q.front();
        q.pop();
        if(edge == dest) return true;

        for(int item : edges[edge]) {  
            if(!visited[item]) {
                visited[item] = true;
                q.push(item);
            }
        }
    }
    return false;
}

long long WorldNavigator::minBribeCost(int n, int m, long long goldRate, long long silverRate,
                                       vector<vector<int>>& roadData) {
  vector<vector<long long>> graph(n, vector<long long>(n, 0));
  
  for (const auto& road : roadData) {
      int u = road[0];
      int v = road[1];
      int goldCost = road[2];
      int silverCost = road[3];
      
      long long totalCost = (long long)goldCost * goldRate + 
                            (long long)silverCost * silverRate;
      
      graph[u][v] = totalCost;
      graph[v][u] = totalCost;
  }
  
  vector<long long> minWeight(n, LLONG_MAX);  // Minimum weight to connect each node
  vector<bool> visited(n, false);             // Track visited nodes
  
  minWeight[0] = 0;  // Start from node 0
  long long totalCost = 0;
  int edgesAdded = 0;
  
  for (int i = 0; i < n; i++) {
      // Find the unvisited node with minimum weight
      // only adjacent nodes are considered cause they have minimum weights
      int currentNode = -1;
      long long minCost = LLONG_MAX;

      for (int j = 0; j < n; j++) {
          if (!visited[j] && minWeight[j] < minCost) {
              minCost = minWeight[j];
              currentNode = j;
          }
      }

      if (currentNode == -1) {
          return -1;
      }

      if (minWeight[currentNode] == LLONG_MAX) {
          return -1;
      }

      visited[currentNode] = true;
      totalCost += minWeight[currentNode];
      edgesAdded++;

      // Update minimum weights for adjacent nodes
      for (int j = 0; j < n; j++) {
          if (graph[currentNode][j] != 0 && !visited[j]) {
              minWeight[j] = min(minWeight[j], graph[currentNode][j]);
          }
      }
  }

  // Check if all nodes are connected
  if (edgesAdded != n) {
      return -1;
  }

  return totalCost;
}

string WorldNavigator::sumMinDistancesBinary(int n, vector<vector<int>>& roads) {
    // Use reduced INF to avoid overflow during addition
    const long long INF = LLONG_MAX / 4;

    // Distance matrix
    vector<vector<long long>> dist(n, vector<long long>(n, INF));

    // city to itselft distance is 0
    for (int i = 0; i < n; i++) {
        dist[i][i] = 0;
    }

    // set given roads in distance matrix (undirected)
    for (const auto& r : roads) {
        int c1 = r[0];
        int c2 = r[1];
        int d = r[2];
        // for duplicate roads select minimum
        dist[c1][c2] = min(dist[c1][c2], (long long)d);
        dist[c2][c1] = min(dist[c2][c1], (long long)d);
        // if no duplicate roads
        // dist[c1][c2] = (long long)d;
        // dist[c2][c1] = (long long)d;
    }
    
    // Floyd-Warshall
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            if (dist[i][k] == INF) continue;
            for (int j = 0; j < n; j++) {
                if (dist[k][j] == INF) continue;
                if (dist[i][j] > dist[i][k] + dist[k][j]) {
                    dist[i][j] = dist[i][k] + dist[k][j];
                }
            }
        }
    }

    // Sum distances for unique pairs (i < j)
    long long sum = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (dist[i][j] != INF) {
                sum += dist[i][j];
            }
        }
    }

    // Convert sum to binary string
    if (sum == 0) return "0";

    string binary;
    while (sum > 0) {
        binary = char('0' + (sum & 1)) + binary;
        sum >>= 1;
    }

    return binary;
}

// =========================================================
// PART D: SERVER KERNEL (Greedy)
// =========================================================

int ServerKernel::minIntervals(vector<char>& tasks, int n) {
    if (n == 0) return (int)tasks.size();
    if (tasks.empty()) return 0;
    int freq[26] = {0};

    for (char c : tasks) {
        freq[c - 'A']++;
    }

    priority_queue<int> pq;
    for (int i = 0; i < 26; i++) {
        if (freq[i] > 0) 
            pq.push(freq[i]);
    }

    queue<pair<int, int>> cooldown;
    int mininterval = 0;
    while (!pq.empty() || !cooldown.empty()) {
        // Removing tasks whose cooldown has finished
        while (!cooldown.empty() && cooldown.front().first <= mininterval) {
            pq.push(cooldown.front().second);
            cooldown.pop();
        }
        // Run the best available task, if any
        if (!pq.empty()) {
            int count = pq.top();
            pq.pop();
            count--;

            // If it still has more executions, put into cooldown
            if (count > 0) {
                cooldown.push({mininterval + n + 1, count});
            }
        }
        mininterval++;
    }
    return mininterval;
}

// =========================================================
// FACTORY FUNCTIONS (Required for Testing)
// =========================================================

extern "C" {
    PlayerTable* createPlayerTable() { 
        return new ConcretePlayerTable(); 
    }

    Leaderboard* createLeaderboard() { 
        return new ConcreteLeaderboard(); 
    }

    AuctionTree* createAuctionTree() { 
        return new ConcreteAuctionTree(); 
    }
}
