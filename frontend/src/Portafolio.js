import React from "react";
import './Portafolio.css';

// Formateador para mostrar como dinero (ej: 100,000.00€)
const currencyFormater = new Intl.NumberFormat('es-ES', {
    style: 'currency',
    currency: 'EUR',
})

const Portafolio = ({portafolio}) => {
    return (
        <div className='portafolio-widget'>
            <h4>Mi Portafolio Virtual</h4>
            <div className='portafolio-cash'>
                <strong>Efectivo:</strong>
                <span>{currencyFormater.format(portafolio.efectivo)}</span>
            </div>
            <div className='portafolio-holdings'>
                <strong>Mis Acciones:</strong>
                {Object.keys(portafolio.posiciones).length === 0 ? (
                    <p className='no-holdings'>No tienes acciones compradas.</p>
                ) : (
                    <ul>
                        {/* Mapeamos el objeto de posiciones */}
                        {Object.entries(portafolio.posiciones).map(([ticker, shares]) => (
                            <li key={ticker}>
                                <span>{ticker}</span>
                                <span>{shares} acciones</span>
                            </li>
                        ))}
                    </ul>
                )}
            </div>
        </div>
    )
}

export default Portafolio;