

// Define the graph nodes
const nodes = ['S', 'A', 'B', 'C', 'D', 'E'];

// Define the edges with weights
const edges = [
  ['S', 'A', 1],
  ['S', 'B', 5],
  ['S', 'C', 10],
  ['S', 'E', -5],
  ['A', 'B', 3],
  ['B', 'C', 2],
  ['D', 'C', 4],
  ['E', 'D', 2]
];

// Initialize distance and predecessor arrays
const dist = {};
const pred = {};

// Set initial values
nodes.forEach(node => {
  dist[node] = Infinity;
  pred[node] = null;
});

// Set source distance to 0
dist['S'] = 0;

// Define the relaxation order
const relaxationOrder = [
  ['S', 'A'], ['B', 'C'], ['S', 'C'], ['S', 'B'],
  ['D', 'C'], ['E', 'D'], ['A', 'B'], ['S', 'E']
];

// Function to relax an edge
function relax(u, v, weight) {
  if (dist[u] !== Infinity && dist[u] + weight < dist[v]) {
    dist[v] = dist[u] + weight;
    pred[v] = u;
    return true; // Edge was relaxed
  }
  return false; // Edge was not relaxed
}

// Track iterations
console.log("Initial distances:", {...dist});

// Perform iterations
for (let k = 1; k <= 2; k++) {
  console.log(`\nIteration ${k}:`);
  
  for (const [u, v] of relaxationOrder) {
    // Find the weight of the edge
    const edge = edges.find(e => e[0] === u && e[1] === v);
    
    if (!edge) {
      console.log(`Edge (${u}, ${v}) not found in the graph`);
      continue;
    }
    
    const weight = edge[2];
    const wasRelaxed = relax(u, v, weight);
    
    console.log(`Relaxing edge (${u}, ${v}) with weight ${weight}: ${wasRelaxed ? 'relaxed' : 'not relaxed'}, dist[${v}] = ${dist[v]}`);
  }
  
  console.log(`\nAfter iteration ${k}, distances:`, {...dist});
}

// Answer: Distance estimate of node C after two iterations
console.log("\nDistance estimate of node C after two iterations:", dist['C']);
Output

Result

