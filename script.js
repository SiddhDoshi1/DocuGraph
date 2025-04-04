// Global variables
let metricsChart = null;
let network = null;
let analyzeNetwork = null;
let centralityChart = null;
let pagerankChart = null;
let coreChart = null;
let currentGraphData = null;

// Document ready
document.addEventListener('DOMContentLoaded', function () {
    // Initialize components
    loadDocuments();
    populateDocumentDropdowns();

    // Event listeners
    document.getElementById('uploadForm').addEventListener('submit', handleUpload);
    document.getElementById('plagiarismForm').addEventListener('submit', handlePlagiarismCheck);
    document.getElementById('compareBtn').addEventListener('click', compareDocuments);
    document.getElementById('files').addEventListener('change', handleFileSelection);

    // Check if analyze tab is active on load
    if (document.querySelector('#analyze-tab.active')) {
        fetchGraphData();
    }

    // Tab change listeners
    document.querySelector('#analyze-tab').addEventListener('shown.bs.tab', function () {
        fetchGraphData();
    });

    // Initial load
    fetchGraphData();
});

// Handle file selection changes
function handleFileSelection(event) {
    const fileList = event.target.files;
    const filePreview = document.getElementById('filePreview');
    filePreview.innerHTML = '';

    if (fileList.length > 0) {
        const list = document.createElement('ul');
        list.className = 'list-group mt-2';

        Array.from(fileList).forEach(file => {
            const item = document.createElement('li');
            item.className = 'list-group-item d-flex justify-content-between align-items-center';
            item.innerHTML = `
                ${file.name} 
                <span class="badge bg-primary rounded-pill">${(file.size / 1024).toFixed(2)} KB</span>
            `;
            list.appendChild(item);
        });

        filePreview.appendChild(list);
    }
}

// Load documents table
async function loadDocuments() {
    try {
        const response = await fetch('http://127.0.0.1:5000/documents');
        console.log(response);
        if (!response.ok) throw new Error('Network response was not ok');

        const docs = await response.json();
        const tbody = document.querySelector('#documentsTable tbody');
        tbody.innerHTML = '';

        docs.forEach(doc => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${doc.filename}</td>
                <td>${new Date(doc.upload_date).toLocaleString()}</td>
                <td>${doc.processed ? 'Yes' : 'No'}</td>
                <td><button class="btn btn-sm btn-danger" onclick="deleteDocument('${doc._id}')">Delete</button></td>
            `;
            tbody.appendChild(row);
        });
    } catch (error) {
        console.error('Error loading documents:', error);
        showAlert('documentsTable', 'Error loading documents', 'danger');
    }
}

// Delete document
async function deleteDocument(docId) {
    if (!confirm('Are you sure you want to delete this document?')) return;

    try {
        const response = await fetch(`http://127.0.0.1:5000/delete_document/${docId}`, {
            method: 'DELETE'
        });

        if (response.ok) {
            loadDocuments();
            populateDocumentDropdowns();
            fetchGraphData();
            showAlert('uploadStatus', 'Document deleted successfully', 'success');
        } else {
            const result = await response.json();
            showAlert('uploadStatus', result.error || 'Error deleting document', 'danger');
        }
    } catch (error) {
        showAlert('uploadStatus', 'Error deleting document: ' + error.message, 'danger');
    }
}

// Handle file upload
async function handleUpload(e) {
    e.preventDefault();
    const statusDiv = document.getElementById('uploadStatus');
    const files = document.getElementById('files').files;

    if (files.length === 0) {
        showAlert('uploadStatus', 'Please select at least one file', 'warning');
        return;
    }

    statusDiv.innerHTML = '<div class="alert alert-info">Uploading files...</div>';

    try {
        const formData = new FormData();
        for (let i = 0; i < files.length; i++) {
            formData.append('files', files[i]);
        }

        const response = await fetch('http://127.0.0.1:5000/upload', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || 'Upload failed');
        }

        // Clear the upload status
        statusDiv.innerHTML = '';

        // Update the documents list immediately
        await loadDocuments();
        await populateDocumentDropdowns();

        // Fetch and display the updated graph metrics
        await fetchGraphData();

        // Show success message
        showAlert('uploadStatus', 'Files uploaded and processed successfully!', 'success');

        // Clear the file input
        document.getElementById('files').value = '';
        document.getElementById('filePreview').innerHTML = '';

    } catch (error) {
        statusDiv.innerHTML = `<div class="alert alert-danger">Error uploading files: ${error.message}</div>`;
    }
}

// Fetch and display graph data with WebSocket for real-time updates
async function fetchGraphData() {
    try {
        showLoading('analyzeGraphContainer', true);

        const response = await fetch('http://127.0.0.1:5000/graph_metrics');
        if (!response.ok) throw new Error('Network response was not ok');

        const data = await response.json();
        console.log("Graph data received:", data);

        if (data && !data.error) {
            currentGraphData = data; // Store current graph data

            // Visualize graph in both containers
            visualizeGraph(data, 'graphContainer');
            visualizeGraph(data, 'analyzeGraphContainer');

            // Display metrics if available
            if (data.metrics) {
                displayGraphMetrics(data.metrics);
            }

            // Initialize WebSocket for real-time updates
            initWebSocket();
        } else {
            console.error("No valid graph data received");
            showAlert('analyzeGraphError', 'No graph data available', 'warning');
        }
    } catch (error) {
        console.error("Error fetching graph data:", error);
        showAlert('analyzeGraphError', 'Error loading graph data', 'danger');
    } finally {
        showLoading('analyzeGraphContainer', false);
    }
}

// Initialize WebSocket connection for real-time updates
function initWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
    const wsUrl = protocol + window.location.host + '/ws';
    const socket = new WebSocket(wsUrl);

    socket.onopen = function () {
        console.log('WebSocket connection established');
    };

    socket.onmessage = function (event) {
        const update = JSON.parse(event.data);
        console.log('WebSocket update received:', update);

        if (update.type === 'graph_update') {
            // Update current graph data
            if (update.data.nodes) {
                currentGraphData.nodes = update.data.nodes;
            }
            if (update.data.edges) {
                currentGraphData.edges = update.data.edges;
            }
            if (update.data.metrics) {
                currentGraphData.metrics = update.data.metrics;
            }

            // Refresh visualizations
            visualizeGraph(currentGraphData, 'graphContainer');
            visualizeGraph(currentGraphData, 'analyzeGraphContainer');
            displayGraphMetrics(currentGraphData.metrics || currentGraphData);
        }
    };

    socket.onerror = function (error) {
        console.error('WebSocket error:', error);
    };

    socket.onclose = function () {
        console.log('WebSocket connection closed');
    };
}

function visualizeGraph(graphData, containerId) {
    const container = document.getElementById(containerId);
    const errorDivId = containerId === 'graphContainer' ? 'graphError' : 'analyzeGraphError';
    const errorDiv = document.getElementById(errorDivId);

    if (!container) {
        console.error(`Container with ID ${containerId} not found`);
        return;
    }

    if (!graphData || !graphData.nodes || !graphData.edges ||
        graphData.nodes.length === 0 || graphData.edges.length === 0) {
        console.error("Invalid or empty graph data");
        if (errorDiv) {
            errorDiv.style.display = 'block';
            errorDiv.textContent = 'No graph data available';
        }
        return;
    }

    if (errorDiv) errorDiv.style.display = 'none';

    try {
        // Create nodes with enhanced properties
        const nodes = new vis.DataSet(
            graphData.nodes.map(node => ({
                id: node.id,
                label: node.label ? node.label.substring(0, 5) : `Doc ${node.id + 1}`,
                title: `Betweenness: ${(node.betweenness || 0).toFixed(4)}\nPageRank: ${(node.pagerank || 0).toFixed(4)}`,
                value: (node.pagerank || 0) * 50,
                color: {
                    background: getColorForPagerank(node.pagerank || 0),
                    border: '#2B7CE9',
                    highlight: {
                        background: getColorForPagerank((node.pagerank || 0) * 1.2),
                        border: '#2B7CE9'
                    }
                },
                font: { size: 16 },
                borderWidth: 2
            }))
        );

        //  edges with weights
        const edges = new vis.DataSet(
            graphData.edges.map(edge => ({
                from: edge.from,
                to: edge.to,
                value: (1.0000-edge.weight || 0) * 5,
                label: `${(edge.weight || 0).toFixed(2)}`, // Display weight with 2 decimal places
                title: `Similarity: ${(edge.weight || 0).toFixed(4)}`,
                color: {
                    color: `rgba(${Math.floor(50 + ((edge.weight || 0)) * 205)}, 
                                 ${Math.floor(50 + ((edge.weight || 0)) * 205)}, 
                                 100, 
                                 ${0.2 + (edge.weight || 0) * 0.8})`,
                    highlight: '#FFA500'
                },
                width: 0.5 + (edge.weight || 0) * 3,
                smooth: { type: 'continuous' },
                font: {
                    size: 12,
                    align: 'top'
                }
            }))
        );

        // Create the network
        const data = { nodes, edges };
            const options = {
                nodes: {
                    scaling: { min: 10, max: 30 }
                },
                edges: {
                    scaling: { min: 1, max: 5 },
                    font: {
                        size: 12,
                        strokeWidth: 3,  // Adds a white border around text for better visibility
                        strokeColor: '#ffffff'
                    },
                    labelHighlightBold: false,
                    smooth: {
                        type: 'continuous',
                        roundness: 0.5
                    }
                },
                physics: {
                    barnesHut: {
                        gravitationalConstant: -2000,
                        centralGravity: 0.3,
                        springLength: 200,
                        springConstant: 0.04,
                        damping: 0.09,
                        avoidOverlap: 0.1
                    },
                    stabilization: { iterations: 2500 }
                },
                layout: { improvedLayout: true },
                interaction: {
                    tooltipDelay: 200,
                    hideEdgesOnDrag: true
                }
            };

        // Destroy previous network instance
        if (containerId === 'graphContainer' && network) {
            network.destroy();
        } else if (containerId === 'analyzeGraphContainer' && analyzeNetwork) {
            analyzeNetwork.destroy();
        }

        // Create new network
        const newNetwork = new vis.Network(container, data, options);

        // Store reference to the correct network
        if (containerId === 'graphContainer') {
            network = newNetwork;
        } else {
            analyzeNetwork = newNetwork;
        }
    } catch (error) {
        console.error("Error visualizing graph:", error);
        if (errorDiv) {
            errorDiv.style.display = 'block';
            errorDiv.textContent = 'Error rendering graph visualization';
        }
    }
}

// Display graph metrics with document-centric centrality
function displayGraphMetrics(metrics) {
    const basicMetricsDiv = document.getElementById('basicMetrics');
    
    // Clear previous content
    basicMetricsDiv.innerHTML = '';
    
    if (!metrics) {
        basicMetricsDiv.innerHTML = '<div class="alert alert-warning">No metrics available</div>';
        return;
    }
    
    // Basic metrics card
    const metricsHTML = `
        <div class="row">
            <div class="col-md-6">
                <div class="metric-card card">
                    <div class="card-body">
                        <h5 class="card-title">Graph Overview</h5>
                        <ul class="list-group list-group-flush">
                            <li class="list-group-item d-flex justify-content-between align-items-center">
                                Documents (Nodes)
                                <span class="badge bg-primary rounded-pill">${metrics.number_of_nodes || 0}</span>
                            </li>
                            <li class="list-group-item d-flex justify-content-between align-items-center">
                                Similarity Links (Edges)
                                <span class="badge bg-primary rounded-pill">${metrics.number_of_edges || 0}</span>
                            </li>
                            <li class="list-group-item d-flex justify-content-between align-items-center">
                                Average Similarity Links
                                <span class="badge bg-primary rounded-pill">${metrics.average_degree?.toFixed(2) || 0}</span>
                            </li>
                            <li class="list-group-item d-flex justify-content-between align-items-center">
                                Graph Density
                                <span class="badge bg-primary rounded-pill">${metrics.density?.toFixed(4) || 0}</span>
                            </li>
                        </ul>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="metric-card card">
                    <div class="card-body">
                        <h5 class="card-title">Top Central Documents</h5>
                        <div class="table-responsive">
                            <table class="table table-sm">
                                <thead>
                                    <tr>
                                        <th>Document</th>
                                        <th>Degree</th>
                                        <th>Betweenness</th>
                                        <th>Closeness</th>
                                    </tr>
                                </thead>
                                <tbody id="centralDocumentsTable">
                                    <!-- Will be filled by JavaScript -->
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    basicMetricsDiv.innerHTML = metricsHTML;
    
    // Fill the central documents table
    populateCentralDocumentsTable(metrics);
    
    // Create centrality charts
    createDocumentCentralityChart(metrics);
    createPagerankChart(metrics);
    createCoreChart(metrics);
}

// Populate the table with top central documents
function populateCentralDocumentsTable(metrics) {
    const tableBody = document.getElementById('centralDocumentsTable');
    tableBody.innerHTML = '';
    
    if (!metrics || !metrics.degree_centrality) return;
    
    // Get all documents with their centrality measures
    const documents = [];
    for (const [docId, degree] of Object.entries(metrics.degree_centrality)) {
        documents.push({
            id: docId,
            name: metrics.doc_name_map[docId] || `Doc ${docId.substring(0, 4)}`,
            degree: degree,
            betweenness: metrics.betweenness_centrality[docId] || 0,
            closeness: metrics.closeness_centrality[docId] || 0,
            eigenvector: metrics.eigenvector_centrality[docId] || 0
        });
    }
    
    // Sort by degree centrality (you could change this to any measure)
    const topDocuments = documents.sort((a, b) => b.degree - a.degree).slice(0, 5);
    
    // Add rows to the table
    topDocuments.forEach(doc => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${doc.name}</td>
            <td>${doc.degree.toFixed(4)}</td>
            <td>${doc.betweenness.toFixed(4)}</td>
            <td>${doc.closeness.toFixed(4)}</td>
        `;
        tableBody.appendChild(row);
    });
}

// Create document centrality chart
function createDocumentCentralityChart(metrics) {
    try {
        const ctx = document.getElementById('centralityChart').getContext('2d');

        if (centralityChart) {
            centralityChart.destroy();
        }

        if (!metrics || !metrics.degree_centrality) {
            console.warn("No centrality metrics available");
            return;
        }

        // Get top 5 documents for each centrality measure
        const topCount = 5;
        const documents = [];
        
        for (const [docId, degree] of Object.entries(metrics.degree_centrality)) {
            documents.push({
                id: docId,
                label: metrics.doc_name_map[docId].substring(0, 5) || `Doc ${docId.substring(0, 4)}`,
                degree: degree,
                betweenness: metrics.betweenness_centrality[docId] || 0,
                closeness: metrics.closeness_centrality[docId] || 0,
                eigenvector: metrics.eigenvector_centrality[docId] || 0
            });
        }

        // Sort and get top documents for each measure
        const byDegree = [...documents].sort((a, b) => b.degree - a.degree).slice(0, topCount);
        const byBetweenness = [...documents].sort((a, b) => b.betweenness - a.betweenness).slice(0, topCount);
        const byCloseness = [...documents].sort((a, b) => b.closeness - a.closeness).slice(0, topCount);
        const byEigenvector = [...documents].sort((a, b) => b.eigenvector - a.eigenvector).slice(0, topCount);

        // Create dataset for chart
        const datasets = [];
        const colors = [
            'rgba(54, 162, 235, 0.7)',
            'rgba(255, 99, 132, 0.7)',
            'rgba(75, 192, 192, 0.7)',
            'rgba(153, 102, 255, 0.7)'
        ];

        // Add a dataset for each centrality measure
        [byDegree, byBetweenness, byCloseness, byEigenvector].forEach((docs, index) => {
            const measure = ['Degree', 'Betweenness', 'Closeness', 'Eigenvector'][index];
            datasets.push({
                label: measure,
                data: docs.map(doc => doc[measure.toLowerCase()]),
                backgroundColor: colors[index],
                borderColor: colors[index].replace('0.7', '1'),
                borderWidth: 1
            });
        });

        centralityChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Top 1', 'Top 2', 'Top 3', 'Top 4', 'Top 5'],
                datasets: datasets
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Top Documents by Centrality Measure'
                    },
                    tooltip: {
                        callbacks: {
                            afterLabel: function(context) {
                                const datasetIndex = context.datasetIndex;
                                const dataIndex = context.dataIndex;
                                let doc;
                                switch(datasetIndex) {
                                    case 0: doc = byDegree[dataIndex]; break;
                                    case 1: doc = byBetweenness[dataIndex]; break;
                                    case 2: doc = byCloseness[dataIndex]; break;
                                    case 3: doc = byEigenvector[dataIndex]; break;
                                }
                                return `Document: ${doc.id.substring(0, 8)}...`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Centrality Value'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Rank Position'
                        }
                    }
                }
            }
        });
    } catch (error) {
        console.error("Error creating centrality chart:", error);
    }
}

// Create pagerank chart
function createPagerankChart(metrics) {
    try {
        const ctx = document.getElementById('pagerankChart').getContext('2d');

        if (pagerankChart) {
            pagerankChart.destroy();
        }

        if (!metrics || !metrics.pagerank || !metrics.doc_name_map) {
            console.warn("No PageRank metrics available");
            return;
        }

        // Get document names from the mapping
        const pagerankEntries = Object.entries(metrics.pagerank)
            .map(([docId, score]) => ({
                id: docId,
                name: metrics.doc_name_map[docId].substring(0, 5) || `Doc ${docId.substring(0,  4)}`,
                score: score
            }))
            .sort((a, b) => b.score - a.score)
            .slice(0, 10);  // Show top 10 documents

        pagerankChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: pagerankEntries.map(item => item.name),  // Use actual document names
                datasets: [{
                    label: 'PageRank Score',
                    data: pagerankEntries.map(item => item.score),
                    backgroundColor: 'rgba(75, 192, 192, 0.7)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1,
                    barPercentage: 0.8,  // Controls bar width
                    categoryPercentage: 0.9  // Controls space between bars
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false, 
                plugins: {
                    title: {
                        display: true,
                        text: 'Top Documents by PageRank'
                    },
                    tooltip: {
                        callbacks: {
                            afterLabel: function(context) {
                                return `Document ID: ${pagerankEntries[context.dataIndex].id}`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'PageRank Score'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Document Name'
                        },
                        ticks: {
                            maxRotation: 45,
                            minRotation: 45
                        }
                    }
                },
                layout: {
                    padding: {
                        left: 10,
                        right: 10,
                        top: 10,
                        bottom: 10
                    }
                }
            }
        });
    } catch (error) {
        console.error("Error creating PageRank chart:", error);
    }
}

// Create core chart
function createCoreChart(metrics) {
    try {
        const ctx = document.getElementById('coreChart').getContext('2d');

        if (coreChart) {
            coreChart.destroy();
        }

        const coreNumbers = metrics.core_numbers || {};
        const coreDistribution = {};

        Object.values(coreNumbers).forEach(core => {
            coreDistribution[core] = (coreDistribution[core] || 0) + 1;
        });

        const sortedCores = Object.keys(coreDistribution).sort((a, b) => a - b);

        coreChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: sortedCores,
                datasets: [{
                    label: 'Nodes per core number',
                    data: sortedCores.map(k => coreDistribution[k]),
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 2,
                    fill: true
                }]
            },
            options: {
                maintainAspectRatio: false, 
                scales: {
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: 'Number of Nodes' }
                    },
                    x: {
                        title: { display: true, text: 'Core Number' }
                    }
                }
            }
        });
    } catch (error) {
        console.error("Error creating core chart:", error);
    }
}

// Helper function to get color based on pagerank
function getColorForPagerank(pagerank) {
    const value = Math.min(255, Math.floor(pagerank * 255 * 10));
    return `rgb(${value}, ${255 - value}, 100)`;
}

// Handle plagiarism check
async function handlePlagiarismCheck(e) {
    e.preventDefault();
    const statusDiv = document.getElementById('plagiarismStatus');
    statusDiv.innerHTML = '<div class="alert alert-info">Checking for plagiarism...</div>';

    const fileInput = document.getElementById('plagiarismFile');
    if (!fileInput.files.length) {
        statusDiv.innerHTML = '<div class="alert alert-warning">Please select a file first</div>';
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    try {
        const response = await fetch('http://127.0.0.1:5000/check_plagiarism', {
            method: 'POST',
            body: formData
        });
        console.log(response);
        const result = await response.json();
        
        if (response.ok) {
            displayPlagiarismResults(result);
        } else {
            statusDiv.innerHTML = `<div class="alert alert-danger">${result.error}</div>`;
        }
    } catch (error) {
        statusDiv.innerHTML = `<div class="alert alert-danger">Error checking for plagiarism: ${error.message}</div>`;
    }
}

// Display plagiarism results
function displayPlagiarismResults(result) {
    const statusDiv = document.getElementById('plagiarismStatus');
    const tbody = document.querySelector('#similarityTable tbody');
    tbody.innerHTML = '';

    if (result.potential_plagiarism) {
        statusDiv.innerHTML = '<div class="alert alert-danger">High similarity detected! Possible plagiarism.</div>';
    } else {
        statusDiv.innerHTML = '<div class="alert alert-success">No significant plagiarism detected.</div>';
    }

    result.results.forEach(res => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${res.document_name.substring(0, 8)}</td>
            <td>${res.similarity_score.toFixed(4)}</td>
            <td>${res.is_plagiarism ?
                '<span class="badge bg-danger">Potential Plagiarism</span>' :
                '<span class="badge bg-success">OK</span>'}</td>
        `;
        tbody.appendChild(row);
    });
}

// Populate document dropdowns
async function populateDocumentDropdowns() {
    try {
        const response = await fetch('http://127.0.0.1:5000/documents');
        if (!response.ok) throw new Error('Network response was not ok');

        const docs = await response.json();
        const doc1Select = document.getElementById('doc1Select');
        const doc2Select = document.getElementById('doc2Select');

        // Clear existing options except the first one
        doc1Select.innerHTML = '<option value="">-- Select Document --</option>';
        doc2Select.innerHTML = '<option value="">-- Select Document --</option>';

        docs.forEach(doc => {
            const option1 = document.createElement('option');
            option1.value = doc._id;
            option1.textContent = doc.filename;
            doc1Select.appendChild(option1);

            const option2 = document.createElement('option');
            option2.value = doc._id;
            option2.textContent = doc.filename;
            doc2Select.appendChild(option2);
        });
    } catch (error) {
        console.error('Error loading documents:', error);
    }
}

// Handle document comparison
async function compareDocuments() {
    const doc1Id = document.getElementById('doc1Select').value;
    const doc2Id = document.getElementById('doc2Select').value;

    if (!doc1Id || !doc2Id) {
        showAlert('pairwiseResults', 'Please select both documents to compare', 'warning');
        return;
    }

    if (doc1Id === doc2Id) {
        showAlert('pairwiseResults', 'Please select two different documents', 'warning');
        return;
    }

    try {
        const response = await fetch(`http://127.0.0.1:5000/analyze_pair?doc_id_1=${doc1Id}&doc_id_2=${doc2Id}`);
        const result = await response.json();

        if (response.ok) {
            displayPairwiseResults(result);
        } else {
            showAlert('pairwiseResults', result.error, 'danger');
        }
    } catch (error) {
        showAlert('pairwiseResults', 'Error comparing documents: ' + error.message, 'danger');
    }
}

// Display pairwise comparison results
function displayPairwiseResults(result) {
    const resultsDiv = document.getElementById('pairwiseResults');
    const similarityScore = result.similarity_score;
    const percentage = (similarityScore * 100).toFixed(2);

    document.getElementById('doc1Name').textContent = result.doc1;
    document.getElementById('doc2Name').textContent = result.doc2;
    document.getElementById('similarityScore').textContent = similarityScore.toFixed(4);

    const similarityBar = document.getElementById('similarityBar');
    similarityBar.style.width = `${percentage}%`;
    similarityBar.textContent = `${percentage}%`;

    // Set color based on similarity level
    if (similarityScore > 0.7) {
        similarityBar.className = 'progress-bar bg-danger';
    } else if (similarityScore > 0.3) {
        similarityBar.className = 'progress-bar bg-warning';
    } else {
        similarityBar.className = 'progress-bar bg-success';
    }

    resultsDiv.style.display = 'block';
}

// Helper functions
function averageObjectValues(obj) {
    const values = Object.values(obj);
    if (values.length === 0) return 0;
    return values.reduce((a, b) => a + b, 0) / values.length;
}

function showAlert(elementId, message, type) {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = `<div class="alert alert-${type}">${message}</div>`;
    } else {
        console.error(`Element with ID ${elementId} not found`);
    }
}

function showLoading(containerId, isLoading) {
    const container = document.getElementById(containerId);
    if (!container) return;

    if (isLoading) {
        container.innerHTML = '<div class="text-center mt-5"><div class="spinner-border text-primary" role="status"><span class="visually-hidden">Loading...</span></div><p>Loading graph data...</p></div>';
    }
}

// Add to global variables
let domainChart = null;

// Add new functions
document.getElementById('classifyBtn').addEventListener('click', async function() {
    const statusDiv = document.getElementById('domainStatus');
    statusDiv.innerHTML = '<div class="alert alert-info">Classifying documents...</div>';
    
    try {
        const response = await fetch('http://127.0.0.1:5000/classify_domains');
        const result = await response.json();
        
        if (response.ok) {
            displayDomainResults(result);
        } else {
            statusDiv.innerHTML = `<div class="alert alert-danger">${result.error || 'Classification failed'}</div>`;
        }
    } catch (error) {
        statusDiv.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
        console.error('Classification error:', error);
    }
});

function displayDomainResults(result) {
    const statusDiv = document.getElementById('domainStatus');
    const tbody = document.querySelector('#domainTable tbody');
    tbody.innerHTML = '';
    
    if (result.results && result.results.length > 0) {
        statusDiv.innerHTML = `<div class="alert alert-success">Classification completed. ${result.results.length} documents analyzed.</div>`;
        
        result.results.forEach(doc => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${doc.document_name.substring(0, 20)}${doc.document_name.length > 20 ? '...' : ''}</td>
                <td><span class="badge bg-primary">${doc.domain}</span></td>
            `;
            tbody.appendChild(row);
        });
        
        // Create domain distribution chart
        createDomainChart(result.domain_distribution);
    } else {
        statusDiv.innerHTML = '<div class="alert alert-warning">No domain classification results available</div>';
    }
}

function createDomainChart(distribution) {
    try {
        const ctx = document.getElementById('domainChart').getContext('2d');
        
        if (domainChart) {
            domainChart.destroy();
        }
        
        const labels = Object.keys(distribution);
        const data = Object.values(distribution);
        const colors = [
            'rgba(255, 99, 132, 0.7)',
            'rgba(54, 162, 235, 0.7)',
            'rgba(255, 206, 86, 0.7)',
            'rgba(75, 192, 192, 0.7)',
            'rgba(153, 102, 255, 0.7)',
            'rgba(255, 159, 64, 0.7)'
        ];
        
        domainChart = new Chart(ctx, {
            type: 'pie',
            data: {
                labels: labels,
                datasets: [{
                    data: data,
                    backgroundColor: colors.slice(0, labels.length),
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Document Domain Distribution'
                    }
                }
            }
        });
    } catch (error) {
        console.error("Error creating domain chart:", error);
    }
}

async function classifyDomainsWithGNN() {
    const statusDiv = document.getElementById('domainStatus');
    statusDiv.innerHTML = '<div class="alert alert-info">Classifying documents with GNN...</div>';
    
    try {
        // First train the GNN
        const trainResponse = await fetch('http://127.0.0.1:5000/train_gnn', {
            method: 'POST'
        });
        
        if (!trainResponse.ok) {
            throw new Error('GNN training failed');
        }
        
        // Then get predictions
        const predictResponse = await fetch('http://127.0.0.1:5000/predict_domains');
        const result = await predictResponse.json();
        
        if (predictResponse.ok) {
            displayDomainResults(result);
        } else {
            statusDiv.innerHTML = `<div class="alert alert-danger">${result.error}</div>`;
        }
    } catch (error) {
        statusDiv.innerHTML = `<div class="alert alert-danger">Error classifying documents: ${error.message}</div>`;
    }
}