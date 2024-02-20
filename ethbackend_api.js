const express = require('express');
const { spawn } = require('child_process');
const bodyParser = require('body-parser');
const app = express();
const port = 3000;

app.use(bodyParser.json());

app.post('/predict', (req, res) => {
  // Extract the user input date from the request body
  const { date } = req.body;

  // Call the Python script to get the prediction
  const pythonProcess = spawn('python', ['ethereumuserinput.py', date]);

  let prediction = null;

  // Collect data from the Python process
  pythonProcess.stdout.on('data', (data) => {
    prediction = data.toString().trim();

    // Send the prediction back to the client
    res.json({ prediction: parseFloat(prediction) });
  });

  // Handle errors and other events if necessary
  pythonProcess.on('error', (error) => {
    console.error(`Error occurred: ${error.message}`);
    res.status(500).json({ error: 'An error occurred while processing the prediction.' });
  });

  pythonProcess.on('close', (code) => {
    console.log(`Python process exited with code ${code}`);
  });
});

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
