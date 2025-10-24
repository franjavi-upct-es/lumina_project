import React, { useState } from "react";
import "./PanelOperar.css";

const PanelOperar = ({ ticker, currentPrice, onComprar, onVender }) => {
  const [cantidad, setCantidad] = useState(1); // Estado para la cantidad

  const handleComprarClick = () => {
    // Validamos que sea un número positivo
    const numCantidad = parseInt(cantidad, 10);
    if (numCantidad > 0) {
      onComprar(cantidad);
      setCantidad(1); // Reseteamos la contidad
    } else {
      alert("Por favor, introduce una cantidad válida.");
    }
  };

  const handleVenderClick = () => {
    const numCantidad = parseInt(cantidad, 10);
    if (numCantidad > 0) {
      onVender(numCantidad);
      setCantidad(1); // Reseteamos la cantidad
    } else {
      alert("Por favor, introduce una cantidad válida.");
    }
  };

  // Calculamos el coste total para mostrar al usuario
  const costeTotal = (currentPrice * (parseInt(cantidad, 10) || 0)).toFixed(2);

  return (
    <div className="trade-widget">
      <h5>Operar {ticker}</h5>
      <p>Precio actual: {currentPrice.toFixed(2)} €</p>
      <div className="trade-controls">
        <label htmlFor="cantidad">Cantidad:</label>
        <input
          type="number"
          id="cantidad"
          value={cantidad}
          onChange={(e) => setCantidad(e.target.value)}
          min="1"
          className="trade-input"
        />
      </div>
      <div className="trade-buttons">
        <button className="trade-button buy" onClick={handleComprarClick}>
          Comprar
        </button>
        <button className="trade-button sell" onClick={handleVenderClick}>
          Vender
        </button>
      </div>
    </div>
  );
};

export default PanelOperar;
