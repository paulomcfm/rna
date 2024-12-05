import React, { useState, useEffect } from 'react';
import { Container, Button, Table, Form, Row, Col } from 'react-bootstrap';
import Papa from 'papaparse';
import 'bootstrap/dist/css/bootstrap.min.css';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

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
    N: 0.1,
    transferFunction: 'linear',
  });
  const [errorsPerEpoch, setErrorsPerEpoch] = useState([]);
  const [chartData, setChartData] = useState({
    labels: [],
    datasets: [
      {
        label: 'Erro por Época',
        data: [],
        borderColor: 'rgba(75,192,192,1)',
        backgroundColor: 'rgba(75,192,192,0.2)',
      },
    ],
  });
  const [finalWeightsHiddenOutput, setFinalWeightsHiddenOutput] = useState([]);
  const [finalWeightsInputHidden, setFinalWeightsInputHidden] = useState([]);
  const [confusionMatrix, setConfusionMatrix] = useState([]);
  const [accuracyGlobal, setAccuracyGlobal] = useState(0);
  const [classAccuracies, setClassAccuracies] = useState([]);


  const updateChartData = (epoch, error) => {
    setErrorsPerEpoch((prevErrors) => {
      const newErrors = [...prevErrors, error];
      const newLabels = newErrors.map((_, index) => index + 1);

      setChartData({
        labels: newLabels,
        datasets: [
          {
            label: 'Erro por Época',
            data: newErrors,
            borderColor: 'rgba(75,192,192,1)',
            backgroundColor: 'rgba(75,192,192,0.2)',
          },
        ],
      });

      return newErrors;
    });
  };


  const handleFileUpload = (file) => {
    Papa.parse(file, {
      header: true,
      complete: (results) => {
        let data = results.data;
        data = data.filter(row => {
          return Object.values(row).some(value => value !== null && value !== '');
        });

        const columnKeys = Object.keys(data[0]);
        const inputLayer = columnKeys.length - 1;
        const classes = [...new Set(data.map(row => row[columnKeys[columnKeys.length - 1]]))];
        const outputLayer = classes.length;
        const hiddenLayer = Math.round((inputLayer + outputLayer) / 2);

        setNeuronsConfig(prevConfig => ({
          ...prevConfig,
          inputLayer,
          outputLayer,
          hiddenLayer,
        }));

        const { normalizedData, params } = normalizeData(data);
        setTrainingData(normalizedData);
        setNormalizationParams(params);
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
        console.log(normalizedData);
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
          normalizedRow[key] = parseFloat(((value - min) / (max - min)).toFixed(4));
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
          normalizedRow[key] = parseFloat(((value - min) / (max - min)).toFixed(4));
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

  const verificaPlato = (errorsPerEpoch, actualPos, n) => {
    const interval = 50;
    if (actualPos >= interval) {
      const initialIntervalPos = actualPos - interval;
      let sum = 0.0;
      let diffSum = 0.0;

      for (let i = initialIntervalPos; i < actualPos; i++) {
        sum += errorsPerEpoch[i];
      }

      const media = sum / interval;

      for (let i = initialIntervalPos; i < actualPos; i++) {
        diffSum += Math.abs(errorsPerEpoch[i] - media);
      }

      const diffMean = diffSum / interval;

      if (diffMean <= n) {
        return true;
      }
    }
    return false;
  };

  const activationFunction = (x) => {
    switch (neuronsConfig.transferFunction) {
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
    y = activationFunction(y);
    switch (neuronsConfig.transferFunction) {
      case 'logistic':
        return y * (1 - y);
      case 'hyperbolic':
        return 1 - (Math.pow(y, 2));
      case 'linear':
      default:
        return 1 / 10;
    }
  };

  const runBackpropagation = () => {
    let backpropagationTrainingData = [];
    if (dataSource === 'subset') {
      const sortedData = [...trainingData].sort((a, b) => {
        if (a.classe < b.classe) return -1;
        if (a.classe > b.classe) return 1;
        return 0;
      });
  
      const newTrainingData = [];
      const newTestData = [];
      let currentClass = sortedData[0]?.classe; 
      let classStartIndex = 0;
  
      sortedData.forEach((row, index) => {
        if (row.classe !== currentClass || index === sortedData.length - 1) {
          const classEndIndex = index === sortedData.length - 1 ? index + 1 : index;
          const classData = sortedData.slice(classStartIndex, classEndIndex);
          const classSize = classData.length;
  
          const trainCount = Math.floor(classSize * 0.7);
          const usedIndices = new Set();
  
          while (usedIndices.size < trainCount) {
            const randomIndex = Math.floor(Math.random() * classSize);
            if (!usedIndices.has(randomIndex)) {
              usedIndices.add(randomIndex);
              newTrainingData.push(classData[randomIndex]);
            }
          }
  
          classData.forEach((item, idx) => {
            if (!usedIndices.has(idx)) {
              newTestData.push(item);
            }
          });
  
          currentClass = row.classe;
          classStartIndex = index;
        }
      });
  
      setTrainingData(newTrainingData);
      setTestData(newTestData);
      backpropagationTrainingData = newTrainingData; 
      console.log('Dados de treino:', newTrainingData);
      console.log('Dados de teste:', newTestData);
    }
    
    const { inputLayer, hiddenLayer, outputLayer, iterations, errorValue, N, transferFunction } = neuronsConfig;
    let learningRate = N;
    const errorsPerEpoch = [];

    const initializeWeights = (rows, cols) => {
      return Array.from({ length: rows }, () =>
        Array.from({ length: cols }, () => Math.random() * 2 - 1)
      );
    };

    let weightsInputHidden = initializeWeights(hiddenLayer, inputLayer);
    let weightsHiddenOutput = initializeWeights(outputLayer, hiddenLayer);

    if (dataSource === 'external') {
      backpropagationTrainingData = trainingData;
    }

    const uniqueClasses = [...new Set(backpropagationTrainingData.map(sample => sample.classe))];
    const classMapping = uniqueClasses.reduce((map, className, index) => {
      map[className] = index;
      return map;
    }, {});

    console.log('Mapeamento de classes:', classMapping);
    let error = 10;
    let outputGradientsFinal = [];
    let shouldContinue = true;
    let shouldIdentifyPlato = true;
    let epochsSinceLastLearningRateChange = 0;

    for (let epoch = 0; epoch < iterations && error > errorValue && shouldContinue; epoch++) {
      let meanError = 0;
      backpropagationTrainingData.forEach((sample) => {
        const inputs = Object.keys(sample)
          .filter(key => key !== 'classe')
          .map(key => parseFloat(sample[key]));

        const desiredOutput = Array(outputLayer).fill(0);
        desiredOutput[classMapping[sample.classe]] = 1;

        const hiddenInputs = weightsInputHidden.map((weights) => {
          return inputs.reduce((sum, input, i) => {
            if (i >= weights.length) {
              return sum;
            }
            return sum + input * weights[i];
          }, 0);
        });
        const hiddenOutputs = hiddenInputs.map(activationFunction);

        const finalInputs = weightsHiddenOutput.map((weights) => {
          return hiddenOutputs.reduce((sum, output, i) => {
            if (i >= weights.length) {
              return sum;
            }
            return sum + output * weights[i];
          }, 0);
        });
        const finalOutputs = finalInputs.map(activationFunction);

        const outputErrors = desiredOutput.map((desired, i) => desired - finalOutputs[i]);

        const outputGradients = outputErrors.map((error, i) => {
          return activationFunctionDerivative(finalInputs[i]) * error;
        });
        outputGradientsFinal = outputGradients;

        const hiddenErrors = weightsHiddenOutput[0].map((_, j) => {
          const sum = outputGradients.reduce((acc, error, k) => {
            return acc + error * weightsHiddenOutput[k][j];
          }, 0);
          return sum * activationFunctionDerivative(hiddenInputs[j]);
        });

        weightsHiddenOutput = weightsHiddenOutput.map((weights, k) =>
          weights.map((weight, j) => weight + learningRate * outputGradients[k] * hiddenOutputs[j])
        );

        weightsInputHidden = weightsInputHidden.map((weights, i) =>
          weights.map((weight, j) => weight + learningRate * hiddenErrors[i] * inputs[j])
        );
      })
      let totalError = 0;
      totalError += outputGradientsFinal.reduce((sum, err) => sum + Math.pow(err, 2), 0);

      meanError += (1 / 2) * totalError;
      console.log('Época', epoch + 1, 'Erro médio:', meanError);
      errorsPerEpoch.push(meanError);
      error = meanError;

      if (shouldIdentifyPlato && epochsSinceLastLearningRateChange >= 50 && verificaPlato(errorsPerEpoch, epoch, errorValue)) {
        console.log('Platô detectado.');
        shouldContinue = window.confirm('Platô detectado. Deseja continuar o treinamento?');
        if (shouldContinue) {
          const modifyLearningRate = window.confirm('Deseja modificar a taxa de aprendizado?');
          if (modifyLearningRate) {
            const newLearningRate = parseFloat(window.prompt('Digite a nova taxa de aprendizado:', learningRate));
            if (!isNaN(newLearningRate) && newLearningRate > 0) {
              learningRate = newLearningRate;
              epochsSinceLastLearningRateChange = 0;
            }
          } else {
            shouldIdentifyPlato = false;
          }
        }
      }
      epochsSinceLastLearningRateChange++;
      updateChartData(epoch, meanError);
    }

    setFinalWeightsHiddenOutput(weightsHiddenOutput);
    setFinalWeightsInputHidden(weightsInputHidden);
    console.log('Treinamento concluído. Média de erros por época:', errorsPerEpoch);
  };

  const runTestData = () => {
    const numClasses = neuronsConfig.outputLayer;
    const confusionMatrix = Array.from({ length: numClasses }, () => Array(numClasses).fill(0));
    let totalTestes = 0;
    let acertos = 0;

    const uniqueClasses = [...new Set(testData.map(sample => sample.classe))];
    const classMapping = uniqueClasses.reduce((map, className, index) => {
      map[className] = index;
      return map;
    }, {});

    testData.forEach((sample) => {
      const inputs = Object.keys(sample)
        .filter(key => key !== 'classe')
        .map(key => parseFloat(sample[key]));

      const desiredClass = classMapping[sample.classe];

      const hiddenInputs = finalWeightsInputHidden.map((weights) => {
        return inputs.reduce((sum, input, i) => sum + input * weights[i], 0);
      });

      const hiddenOutputs = hiddenInputs.map(activationFunction);

      const finalInputs = finalWeightsHiddenOutput.map((weights) => {
        return hiddenOutputs.reduce((sum, output, i) => sum + output * weights[i], 0);
      });

      const finalOutputs = finalInputs.map(activationFunction);

      let maxOutput = finalOutputs[0];
      let identifiedClass = 0;

      for (let k = 1; k < finalOutputs.length; k++) {
        if (finalOutputs[k] > maxOutput) {
          maxOutput = finalOutputs[k];
          identifiedClass = k;
        }
      }

      confusionMatrix[desiredClass][identifiedClass]++;
      totalTestes++;
      if (desiredClass === identifiedClass) {
        acertos++;
      }
    });

    const acuraciaGlobal = (acertos / totalTestes) * 100;

    const acuraciaPorClasse = confusionMatrix.map((row, i) => {
      const totalClasse = row.reduce((sum, value) => sum + value, 0);
      const acertosClasse = row[i];
      return totalClasse > 0 ? (acertosClasse / totalClasse) * 100 : 0;
    });

    acuraciaPorClasse.forEach((acuracia, i) => {
      console.log(`Acurácia da classe ${i + 1}:`, acuracia, '%');
    });
    setConfusionMatrix(confusionMatrix);
    setAccuracyGlobal(acuraciaGlobal);
    setClassAccuracies(acuraciaPorClasse);

    console.log('Confusion Matrix:', confusionMatrix);
    console.log('Acurácia Global:', acuraciaGlobal, '%');

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
                  setNeuronsConfig({ ...neuronsConfig, hiddenLayer: parseInt(e.target.value, 10) })
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
            <option value="external">Use external file as testing dataset</option>
            <option value="subset">Use a subset of the original file</option>
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
          <h1>Traning Data</h1>
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
      {testData.length > 0 && (
        <div style={{ maxHeight: '400px', overflowY: 'scroll' }}>
          <h1>Traning Data</h1>
          <Table striped bordered hover className="mt-3">
            <thead>
              <tr>
                {Object.keys(testData[0]).map((header, index) => (
                  <th key={index}>{header}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {testData.map((row, rowIndex) => (
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
      {trainingData.length > 0 && testData.length > 0 && dataSource === 'external' && (
        <Button className="mt-3" onClick={runBackpropagation}>
          Solve
        </Button>
      )}
      {trainingData.length > 0 && dataSource === 'subset' && (
        <Button className="mt-3" onClick={runBackpropagation}>
          Solve
        </Button>
      )}
      {chartData.labels.length > 0 && (
        <Line
          data={chartData}
          options={{
            responsive: false,
            maintainAspectRatio: false,
            plugins: {
              tooltip: {
                callbacks: {
                  label: function (context) {
                    return `Erro: ${context.raw}`;
                  },
                },
              },
            },
            scales: {
              y: {
                ticks: {
                  callback: function (value) {
                    return value;
                  },
                },
              },
            },
          }}
          width={800}
          height={400}
        />
      )}
      {chartData.labels.length > 0 && (
        <Button className="mt-3" onClick={runTestData}>
          Test
        </Button>
      )}
      {confusionMatrix.length > 0 && (
        <div className="mt-5">
          <h3>Confusion Matrix</h3>
          <Table striped bordered hover>
            <thead>
              <tr>
                <th>Real Class</th>
                {confusionMatrix.map((_, index) => (
                  <th key={index}>Classe {index + 1}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {confusionMatrix.map((row, i) => (
                <tr key={i}>
                  <td>Class {i + 1}</td>
                  {row.map((value, j) => (
                    <td key={j}>{value}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </Table>
          <h4>Global Accuracy: {accuracyGlobal.toFixed(2)}%</h4>
          <h4>Accuracy per Class:</h4>
          <ul>
            {classAccuracies.map((acc, index) => (
              <li key={index}>Class {index + 1}: {acc.toFixed(2)}%</li>
            ))}
          </ul>
        </div>
      )}
    </Container>
  );
}

export default App;
