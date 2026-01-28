const express = require('express');
const axios = require('axios');
const https = require('https');
const path = require('path');
const PDFDocument = require('pdfkit');

const app = express();
app.use(express.json());

// Create a shared HTTPS agent with TLS 1.2
const agent = new https.Agent({
  rejectUnauthorized: true, // verify SSL certificate
  minVersion: 'TLSv1.2'
});

// Serve HTML
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

// Fetch Project Report JSON
app.get('/generate-full-report/:projectId', async (req, res) => {
  const projectId = req.params.projectId;
  console.log(`Project ID received: ${projectId}`);
  const url = `https://oracleapex.com/ords/ueacmp/project_report/project/${projectId}`;
  console.log(`Fetching Project Report URL: ${url}`);

  try {
    const response = await axios.get(url, {
      httpsAgent: agent,
      headers: { 'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json' }
    });
    res.json(response.data);
  } catch (err) {
    console.error('ORDS fetch error (project report):', err.message);
    res.status(500).json({ error: 'Failed to fetch data', details: err.message });
  }
});

// Fetch Control Doc JSON
app.get('/generate-control-doc/:projectId', async (req, res) => {
  const projectId = req.params.projectId;
  const url = `https://oracleapex.com/ords/ueacmp/control_doc/project/${projectId}`;
  console.log(`Fetching Control Doc URL: ${url}`);

  try {
    const response = await axios.get(url, {
      httpsAgent: agent,
      headers: { 'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json' }
    });
    res.json(response.data);
  } catch (err) {
    console.error('ORDS fetch error (control doc):', err.message);
    res.status(500).json({ error: 'Failed to fetch data', details: err.message });
  }
});

// Hazard Identification PDF
app.get('/download-hazard-report/:projectId', async (req, res) => {
  const projectId = req.params.projectId;

  try {
    // Fetch the full report JSON
    const response = await axios.get(`http://localhost:3000/generate-full-report/${projectId}`);
    const data = response.data;

    // Create PDF
    const doc = new PDFDocument({ margin: 30, size: 'A4' });

    // Set headers for download
    res.setHeader('Content-disposition', `attachment; filename=ProjectReport_${projectId}.pdf`);
    res.setHeader('Content-type', 'application/pdf');

    doc.pipe(res);

    // PDF content
    doc.fontSize(18).text(`Project Report - ID: ${projectId}`, { align: 'center' });
    doc.moveDown();

    data.forEach((hazard, index) => {
      doc.fontSize(14).text(`Hazard ${index + 1}: ${hazard.hazard_code} - ${hazard.hazard_detail}`);
      doc.fontSize(12)
        .text(`Initial Risk: ${hazard.initial_risk_level}`)
        .text(`Post Risk: ${hazard.post_risk_level}`)
        .text(`ALARP Statement: ${hazard.alarp_statement}`)
        .text(`Hazard Status: ${hazard.hazard_status}`)
        .text(`Accidents: ${hazard.accidents || 'None'}`)
        .text(`Causes: ${hazard.causes || 'None'}`)
        .text(`Controls: ${hazard.controls || 'None'}`)
        .text(`References: ${hazard.references || 'None'}`)
        .moveDown();
    });

    doc.end();

  } catch (err) {
    console.error('PDF generation error:', err);
    res.status(500).send('Failed to generate PDF');
  }
});

// Control Identification PDF
app.get('/download-control-report/:projectId', async (req, res) => {
  const projectId = req.params.projectId;

  try {
    // Fetch the full report JSON
    const response = await axios.get(`http://localhost:3000/generate-full-report/${projectId}`);
    const data = response.data;

    // Create PDF
    const doc = new PDFDocument({ margin: 30, size: 'A4' });

    // Set headers for download
    res.setHeader('Content-disposition', `attachment; filename=ProjectReport_${projectId}.pdf`);
    res.setHeader('Content-type', 'application/pdf');

    doc.pipe(res);

    // PDF content
    doc.fontSize(18).text(`Project Report - ID: ${projectId}`, { align: 'center' });
    doc.moveDown();

    data.forEach((hazard, index) => {
      doc.fontSize(14).text(`Hazard ${index + 1}: ${hazard.hazard_code} - ${hazard.hazard_detail}`);
      doc.fontSize(12)
        .text(`Initial Risk: ${hazard.initial_risk_level}`)
        .text(`Post Risk: ${hazard.post_risk_level}`)
        .text(`ALARP Statement: ${hazard.alarp_statement}`)
        .text(`Hazard Status: ${hazard.hazard_status}`)
        .text(`Accidents: ${hazard.accidents || 'None'}`)
        .text(`Causes: ${hazard.causes || 'None'}`)
        .text(`Controls: ${hazard.controls || 'None'}`)
        .text(`References: ${hazard.references || 'None'}`)
        .moveDown();
    });

    doc.end();

  } catch (err) {
    console.error('PDF generation error:', err);
    res.status(500).send('Failed to generate PDF');
  }
});

// Node server port
const PORT = 3000;
app.listen(PORT, () => console.log(`Generator service running on port ${PORT}`));
