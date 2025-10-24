import React from "react";
// Importamos los componentes necesarios de Chart.js
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

// Registramos los componentes de Chart.js que vamos a usar
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
);

// Este es el componente que dibuja el gráfico
const StockChart = ({ chartData, companyName }) => {
  const SHORT_SMA_KEY = "SMA_50";
  const LONG_SMA_KEY = "SMA_200";

  // 1. Preparamos los datos para el gráfico
  const data = {
    // Las etiquetas de eje X (las fechas)
    labels: chartData.map((d) => d.Date),
    datasets: [
      {
        label: `Precio de Cierre de ${companyName}`,
        data: chartData.map((d) => d.Close),
        borderColor: "rgb(75, 192, 192)", // Color de la línea
        tension: 0.1, // Una ligera curva en la línea
        pointRadius: 0, // No mostrar puntos en cada fecha
      },
      {
        label: `${SHORT_SMA_KEY} (${companyName})`,
        data: chartData.map((d) => d[SHORT_SMA_KEY]), // Usamos la clave dinámica
        borderColor: "rgb(255, 99, 132)", // Un color rojo/rosa para la rápida
        backgroundColor: "rgba(255, 99, 132, 0.5)",
        tension: 0.1,
        pointRadius: 0,
        borderDash: [5, 5], // Hacemos la línea de puntos para diferenciarla
      },
      {
        label: `${LONG_SMA_KEY} (${companyName})`,
        data: chartData.map((d) => d[LONG_SMA_KEY]), // Usamos la clave dinámica
        borderColor: "rgb(54, 162, 235)", // Un color azul para la lenta
        backgroundColor: "rgba(54, 162, 235, 0.5)",
        tension: 0.1,
        pointRadius: 0,
        borderDash: [5, 5], // Hacemos la línea de puntos para diferenciarla
      },
    ],
  };

  // 2. Configuramos las opciones del gráfico
  const options = {
    responsive: true, // Hace que el gráfico se adapte al tamaño del contenedor
    interaction: {
      mode: "index", // Esto permite que el tooltip muestre datos de múltiples líneas
      intersect: false,
    },
    plugins: {
      legend: {
        position: "top", // La leyenda (ej: "Precio de Cierre...") arriba
      },
      title: {
        display: true,
        text: `Historial de Precios de 1 Año`,
      },
      tooltip: {
        mode: "index",
        intersect: false,
      },
    },
    scales: {
      x: {
        display: false, // Ocultamos las etiquetas del eje X (son demasiadas)
      },
      y: {
        title: {
          display: true,
          text: "Precio (USD)",
        },
      },
    },
  };

  // 3. Devolvemos el componente del gráfico
  return <Line data={data} options={options} />;
};

export default StockChart;
