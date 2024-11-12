import React, { useState } from 'react';
import { Container, Button, Table, Form, Row, Col } from 'react-bootstrap';
import Papa from 'papaparse';
import 'bootstrap/dist/css/bootstrap.min.css';
import { Line } from 'react-chartjs-2';

function App() {
  const [trainingData, setTrainingData] = useState([]);
  const [testData, setTestData] = useState([]);
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
        let data = results.data;
        data = data.filter(row => {
          return Object.values(row).some(value => value !== null && value !== '');
        });
        const { normalizedData, params } = normalizeData(data);
        setTrainingData(normalizedData);
        setNormalizationParams(params);
        console.log(data);
      },
    });
    setDataSource('subset');
  };

  const handleExternalFileUpload = (file) => {
    Papa.parse(file, {
      header: true,
      complete: (results) => {
        let data = results.data;
        data = data.filter(row => {
          return Object.values(row).some(value => value !== null && value !== '');
        });
        const normalizedData = normalizeExternalData(data, normalizationParams);
        setTestData(normalizedData);
        console.log(data);
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
    if (dataSource === 'subset') {
      const classCounts = {};
      const classData = {};

      trainingData.forEach((row) => {
        const className = row[row.length - 1];
        if (!classCounts[className]) {
          classCounts[className] = 0;
          classData[className] = [];
        }
        classCounts[className]++;
        classData[className].push(row);
      });

      const newTrainingData = [];
      const newTestData = [];

      Object.keys(classData).forEach((className) => {
        const data = classData[className];
        const testSize = Math.floor(data.length * 0.3);
        const trainingSize = data.length - testSize;

        newTrainingData.push(...data.slice(0, trainingSize));

        newTestData.push(...data.slice(trainingSize));
      });

      setTrainingData(newTrainingData);
      setTestData(newTestData);
      console.log(newTrainingData);
      console.log(newTestData);
    }

    const { inputLayer, hiddenLayer, outputLayer, iterations, errorValue, N, transferFunction } = neuronsConfig;
    const learningRate = N;
    const errorsPerEpoch = [];

    // Inicializar pesos aleatórios entre -1 e 1
    const initializeWeights = (rows, cols) => {
      return Array.from({ length: rows }, () =>
        Array.from({ length: cols }, () => Math.random() * 2 - 1)
      );
    };

    let weightsInputHidden = initializeWeights(inputLayer, hiddenLayer);
    let weightsHiddenOutput = initializeWeights(hiddenLayer, outputLayer);

    const activationFunction = (x) => {
      switch (transferFunction) {
        case 'logistic':
          return 1 / (1 + Math.exp(-x));
        case 'hyperbolic':
          return Math.tanh(x);
        case 'linear':
        default:
          return x / 10;
      }
    };

    const activationFunctionDerivative = (y) => {
      switch (transferFunction) {
        case 'logistic':
          return y * (1 - y);
        case 'hyperbolic':
          return 1 - Math.pow(y, 2);
        case 'linear':
        default:
          return 1 / 10;
      }
    };

    // Gerar mapeamento de classes dinâmico
    const uniqueClasses = [...new Set(trainingData.map(sample => sample.classe))];
    const classMapping = uniqueClasses.reduce((map, className, index) => {
      map[className] = index;
      return map;
    }, {});

    console.log('Mapeamento de classes:', classMapping);
    let error = 10;
    for (let epoch = 0; epoch < iterations || error > errorValue; epoch++) {
      let totalError = 0;

      trainingData.forEach((sample) => {
        // Separar entrada e saída dinamicamente
        const inputs = Object.keys(sample)
          .filter(key => key !== 'classe') 
          .map(key => parseFloat(sample[key]));

        const desiredOutput = Array(outputLayer).fill(0);
        desiredOutput[classMapping[sample.classe]] = 1;

        // Forward pass
        const hiddenInputs = weightsInputHidden.map((weights) => {
          return inputs.reduce((sum, input, i) => {
            if (i >= weights.length) {
              return sum;
            }
            return sum + input * weights[i];
          }, 0);
        });
        const hiddenOutputs = hiddenInputs.map(activationFunction);

        const finalInputs = weightsHiddenOutput.map((weights) =>{
          return hiddenOutputs.reduce((sum, output, i) => {
            if (i >= weights.length) {
              return sum;
            }
            return sum + output * weights[i];
          });
        });
        const finalOutputs = finalInputs.map(activationFunction);

        // Calcular erro
        const outputErrors = desiredOutput.map((desired, i) => desired - finalOutputs[i]);
        totalError += outputErrors.reduce((sum, err) => sum + Math.pow(err, 2), 0);

        // Backward pass (erro da camada de saída)
        const outputGradients = outputErrors.map((error, i) =>
          error * activationFunctionDerivative(finalOutputs[i])
        );

        // Erro da camada oculta
        const hiddenErrors = weightsHiddenOutput[0].map((_, j) =>
          outputGradients.reduce((sum, grad, k) => sum + grad * weightsHiddenOutput[k][j], 0)
        );
        const hiddenGradients = hiddenErrors.map((error, i) =>
          error * activationFunctionDerivative(hiddenOutputs[i])
        );

        // Atualizar pesos entre camada oculta e de saída
        weightsHiddenOutput = weightsHiddenOutput.map((weights, k) =>
          weights.map((weight, j) => weight + learningRate * outputGradients[k] * hiddenOutputs[j])
        );

        // Atualizar pesos entre entrada e camada oculta
        weightsInputHidden = weightsInputHidden.map((weights, i) =>
          weights.map((weight, j) => weight + learningRate * hiddenGradients[i] * inputs[j])
        );
      });

      // Armazenar a média de erros desta época
      const meanError = totalError / trainingData.length;
      errorsPerEpoch.push(meanError);
      console.log('Época', epoch + 1, 'Erro médio:', meanError);
      error = meanError;

      // Atualizar dados do gráfico
      setErrorData((prevData) => ({
        ...prevData,
        labels: [...prevData.labels, epoch + 1],
        datasets: [{
          ...prevData.datasets[0],
          data: [...prevData.datasets[0].data, meanError],
        }],
      }));
    }

    console.log('Treinamento concluído. Média de erros por época:', errorsPerEpoch);
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

      {trainingData.length > 0 && (
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
      {trainingData.length > 0 && (
        <div style={{ maxHeight: '400px', overflowY: 'scroll' }}>
          <Table striped bordered hover className="mt-3">
            <thead>
              <tr>
                {Object.keys(trainingData[0]).map((header, index) => (
                  <th key={index}>{header}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {trainingData.map((row, rowIndex) => (
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
      {trainingData.length > 0 && (
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
