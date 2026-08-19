import jsPDF from "jspdf";
import html2canvas from "html2canvas";

function ExportReport({ elementId }) {

  const exportReport = async () => {

    const element =
      document.getElementById(elementId);

    if (!element) return;

    const canvas =
      await html2canvas(element);

    const image =
      canvas.toDataURL("image/png");

    const pdf =
      new jsPDF("p", "mm", "a4");

    const width = 190;

    const height =
      (canvas.height * width) /
      canvas.width;

    pdf.addImage(
      image,
      "PNG",
      10,
      10,
      width,
      height
    );

    pdf.save("sdoh-risk-report.pdf");
  };

  return (
    <button onClick={exportReport}>
      Export Report
    </button>
  );
}

export default ExportReport;