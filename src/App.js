import React, { useState } from 'react';
import { Container, Button, Table, Form, Row, Col } from 'react-bootstrap';
import Papa from 'papaparse';
import 'bootstrap/dist/css/bootstrap.min.css';
import { Line } from 'react-chartjs-2';

function App() {
  const [csvData, setCsvData] = useState([]);
  const [dataSource, setDataSource] = useState('');
  const [normalizationParams, setNormalizationParams] = useState({});
  const [neuronsConfig, setNeuronsConfig] = useState({
    inputLayer: 6,
    outputLayer: 5,
    hiddenLayer: 5,
    errorValue: 0.00001,
    iterations: 2000,
    N: 0.2,
    transferFunction: 'linear',
  });
  const [errorData, setErrorData] = useState({ labels: [], datasets: [{ label: 'Mean Error', data: [] }] });

  const handleFileUpload = (file) => {
    Papa.parse(file, {
      header: true,
      complete: (results) => {
        const data = results.data;
        const { normalizedData, params } = normalizeData(data);
        setCsvData(normalizedData);
        setNormalizationParams(params);
      },
    });
    setDataSource('external');
  };

  const handleExternalFileUpload = (file) => {
    Papa.parse(file, {
      header: true,
      complete: (results) => {
        const data = results.data;
        const normalizedData = normalizeExternalData(data, normalizationParams);
        console.log('Normalized external file data:', normalizedData);
      },
    });
  };

  const normalizeExternalData = (data, params) => {
    const columnKeys = Object.keys(data[0]);
  
    const normalizedData = data.map((row) => {
      const normalizedRow = {};
      columnKeys.forEach((key) => {
        const value = parseFloat(row[key]);
        if (!isNaN(value) && params[key]) {
          const { min, max } = params[key];
          normalizedRow[key] = (value - min) / (max - min);
        } else {
          normalizedRow[key] = row[key]; 
        }
      });
      return normalizedRow;
    });
  
    return normalizedData;
  };

  const normalizeData = (data) => {
    const params = {};
    const columnKeys = Object.keys(data[0]);
  
    data.forEach((row) => {
      columnKeys.forEach((key) => {
        const value = parseFloat(row[key]);
        if (!isNaN(value)) {
          if (!params[key]) {
            params[key] = { min: value, max: value };
          } else {
            if (value < params[key].min) params[key].min = value;
            if (value > params[key].max) params[key].max = value;
          }
        }
      });
    });
  
    const normalizedData = data.map((row) => {
      const normalizedRow = {};
      columnKeys.forEach((key) => {
        const value = parseFloat(row[key]);
        if (!isNaN(value)) {
          const { min, max } = params[key];
          normalizedRow[key] = (value - min) / (max - min);
        } else {
          normalizedRow[key] = row[key];
        }
      });
      return normalizedRow;
    });
  
    return { normalizedData, params };
  };

  const handleNChange = (e) => {
    let value = parseFloat(e.target.value);
    if (value <= 0) {
      value = 0.1;
    } else if (value > 1) {
      value = 1;
    }
    setNeuronsConfig({ ...neuronsConfig, N: value });
  };

  const runBackpropagation = () => {
    // Placeholder for backpropagation algorithm
    const iterations = 100;
    const errors = [];
    for (let i = 0; i < iterations; i++) {
      // Simulate error calculation
      const meanError = Math.random(); // Replace with actual error calculation
      errors.push(meanError);
    }
    setErrorData({
      labels: Array.from({ length: iterations }, (_, i) => i + 1),
      datasets: [{ label: 'Mean Error', data: errors }]
    });
  };

  return (
    <Container className="mt-5">
      <h2>Neural Network Configuration</h2>
      <Form>
        <Row>
          <Col>
            <Form.Group>
              <Form.Label>Input Layer Neurons</Form.Label>
              <Form.Control
                type="number"
                value={neuronsConfig.inputLayer}
                readOnly
              />
            </Form.Group>
            <Form.Group>
              <Form.Label>Output Layer Neurons</Form.Label>
              <Form.Control
                type="number"
                value={neuronsConfig.outputLayer}
                onChange={(e) =>
                  setNeuronsConfig({ ...neuronsConfig, outputLayer: e.target.value })
                }
                readOnly
              />
            </Form.Group>
            <Form.Group>
              <Form.Label>Hidden Layer Neurons</Form.Label>
              <Form.Control
                type="number"
                value={neuronsConfig.hiddenLayer}
                onChange={(e) =>
                  setNeuronsConfig({ ...neuronsConfig, hiddenLayer: e.target.value })
                }
              />
            </Form.Group>
          </Col>
          <Col>
            <Form.Group>
              <Form.Label>Error Value</Form.Label>
              <Form.Control
                type="number"
                step="0.00001"
                value={neuronsConfig.errorValue}
                onChange={(e) =>
                  setNeuronsConfig({ ...neuronsConfig, errorValue: e.target.value })
                }
              />
            </Form.Group>
            <Form.Group>
              <Form.Label>Number of Iterations</Form.Label>
              <Form.Control
                type="number"
                value={neuronsConfig.iterations}
                onChange={(e) =>
                  setNeuronsConfig({ ...neuronsConfig, iterations: e.target.value })
                }
              />
            </Form.Group>
          </Col>
          <Col>
            <Form.Group>
              <Form.Label>N</Form.Label>
              <Form.Control
                type="number"
                step="0.01"
                min="0.01"
                max="1"
                value={neuronsConfig.N}
                onChange={handleNChange}
              />
            </Form.Group>
          </Col>
          <Col>
            <Form.Group>
              <Form.Label>Transfer Function</Form.Label>
              <Form.Control
                as="select"
                value={neuronsConfig.transferFunction}
                onChange={(e) =>
                  setNeuronsConfig({ ...neuronsConfig, transferFunction: e.target.value })
                }
              >
                <option value="linear">Linear</option>
                <option value="logistic">Logistic</option>
                <option value="hyperbolic">Hyperbolic</option>
              </Form.Control>
            </Form.Group>
          </Col>
        </Row>
      </Form>

      <hr />
      <h3>Upload CSV to View Data</h3>
      <Form.Group>
        <Form.Label>Choose CSV File</Form.Label>
        <Form.Control
          type="file"
          onChange={(e) => handleFileUpload(e.target.files[0])}
          accept=".csv"
        />
      </Form.Group>

      {csvData.length > 0 && (
        <Form.Group>
          <Form.Label>Choose Data Source</Form.Label>
          <Form.Control
            as="select"
            value={dataSource}
            onChange={(e) => setDataSource(e.target.value)}
          >
            <option value="external">Utilizar um arquivo externo para teste</option>
            <option value="subset">Utilizar um subconjunto do arquivo original</option>
          </Form.Control>
        </Form.Group>
      )}
      {dataSource === 'external' && (
          <Form.Group>
            <Form.Label>Upload External CSV File</Form.Label>
            <Form.Control
              type="file"
              onChange={(e) => handleExternalFileUpload(e.target.files[0])}
              accept=".csv"
            />
          </Form.Group>
        )}
      {csvData.length > 0 && (
        <div style={{ maxHeight: '400px', overflowY: 'scroll' }}>
          <Table striped bordered hover className="mt-3">
            <thead>
              <tr>
                {Object.keys(csvData[0]).map((header, index) => (
                  <th key={index}>{header}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {csvData.map((row, rowIndex) => (
                <tr key={rowIndex}>
                  {Object.values(row).map((value, colIndex) => (
                    <td key={colIndex}>{value}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </Table>
        </div>
      )}
      {csvData.length > 0 && (
        <Button className="mt-3" onClick={runBackpropagation}>
          Resolver
        </Button>
      )}

      {errorData.labels.length > 0 && (
        <div className="mt-3">
          <h4>Mean Error per Iteration</h4>
          <Line data={errorData} />
        </div>
      )}
    </Container>
  );
}

export default App;
