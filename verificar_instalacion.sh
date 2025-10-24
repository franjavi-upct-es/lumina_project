#!/bin/bash

# Script de verificación pre-ejecución
# Ejecuta: bash verificar_instalacion.sh

echo "🔍 VERIFICANDO INSTALACIÓN DE LUMINA v2.0"
echo "=========================================="
echo ""

# Colores
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Contadores
total_checks=0
passed_checks=0

# Función para verificar
check() {
    total_checks=$((total_checks + 1))
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅${NC} $2"
        passed_checks=$((passed_checks + 1))
    else
        echo -e "${RED}❌${NC} $2"
    fi
}

# 1. Verificar Python
echo "📦 Verificando Backend..."
python3 --version > /dev/null 2>&1
check $? "Python 3 instalado"

# 2. Verificar pip
pip3 --version > /dev/null 2>&1
check $? "pip instalado"

# 3. Verificar archivo requirements.txt
if [ -f "backend/requirements.txt" ]; then
    check 0 "requirements.txt existe"
else
    check 1 "requirements.txt existe"
fi

# 4. Verificar archivos de servicio
if [ -f "backend/news_service.py" ]; then
    check 0 "news_service.py creado"
else
    check 1 "news_service.py creado"
fi

if [ -f "backend/lstm_service.py" ]; then
    check 0 "lstm_service.py creado"
else
    check 1 "lstm_service.py creado"
fi

# 5. Verificar .env.example
if [ -f "backend/.env.example" ]; then
    check 0 ".env.example creado"
else
    check 1 ".env.example creado"
fi

echo ""
echo "🎨 Verificando Frontend..."

# 6. Verificar Node.js
node --version > /dev/null 2>&1
check $? "Node.js instalado"

# 7. Verificar npm
npm --version > /dev/null 2>&1
check $? "npm instalado"

# 8. Verificar componentes nuevos
if [ -f "frontend/src/LSTMPredictor.js" ]; then
    check 0 "LSTMPredictor.js creado"
else
    check 1 "LSTMPredictor.js creado"
fi

if [ -f "frontend/src/NewsPanel.js" ]; then
    check 0 "NewsPanel.js creado"
else
    check 1 "NewsPanel.js creado"
fi

# 9. Verificar archivos CSS
if [ -f "frontend/src/LSTMPredictor.css" ]; then
    check 0 "LSTMPredictor.css creado"
else
    check 1 "LSTMPredictor.css creado"
fi

if [ -f "frontend/src/NewsPanel.css" ]; then
    check 0 "NewsPanel.css creado"
else
    check 1 "NewsPanel.css creado"
fi

echo ""
echo "📚 Verificando Documentación..."

# 10. Verificar documentación
if [ -f "NUEVAS_FUNCIONALIDADES.md" ]; then
    check 0 "NUEVAS_FUNCIONALIDADES.md creado"
else
    check 1 "NUEVAS_FUNCIONALIDADES.md creado"
fi

if [ -f "IMPLEMENTACION_COMPLETADA.md" ]; then
    check 0 "IMPLEMENTACION_COMPLETADA.md creado"
else
    check 1 "IMPLEMENTACION_COMPLETADA.md creado"
fi

if [ -f "RESUMEN_EJECUTIVO.md" ]; then
    check 0 "RESUMEN_EJECUTIVO.md creado"
else
    check 1 "RESUMEN_EJECUTIVO.md creado"
fi

if [ -f "test_new_features.py" ]; then
    check 0 "test_new_features.py creado"
else
    check 1 "test_new_features.py creado"
fi

echo ""
echo "🔐 Verificando Configuración..."

# 11. Verificar .gitignore
if [ -f "backend/.gitignore" ]; then
    check 0 ".gitignore creado"
else
    check 1 ".gitignore creado"
fi

# 12. Verificar estructura de directorios
if [ -d "backend" ]; then
    check 0 "Directorio backend existe"
else
    check 1 "Directorio backend existe"
fi

if [ -d "frontend/src" ]; then
    check 0 "Directorio frontend/src existe"
else
    check 1 "Directorio frontend/src existe"
fi

echo ""
echo "🧪 Verificando Dependencias Python..."

# Verificar si está en venv
if [ -d "backend/.venv" ] || [ -d ".venv" ]; then
    echo -e "${GREEN}✅${NC} Entorno virtual encontrado"
    passed_checks=$((passed_checks + 1))
    
    # Activar venv y verificar paquetes
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    elif [ -f "backend/.venv/bin/activate" ]; then
        source backend/.venv/bin/activate
    fi
    
    # Verificar paquetes críticos
    pip list 2>/dev/null | grep -q "newsapi-python"
    check $? "newsapi-python instalado"
    
    pip list 2>/dev/null | grep -q "tensorflow"
    check $? "tensorflow instalado"
    
    pip list 2>/dev/null | grep -q "scikit-learn"
    check $? "scikit-learn instalado"
    
    pip list 2>/dev/null | grep -q "python-dotenv"
    check $? "python-dotenv instalado"
    
    total_checks=$((total_checks + 4))
else
    echo -e "${YELLOW}⚠️${NC}  Entorno virtual no encontrado"
    echo "   Ejecuta: cd backend && python -m venv .venv"
    total_checks=$((total_checks + 5))
fi

echo ""
echo "=========================================="
echo -e "📊 RESULTADO: ${GREEN}${passed_checks}${NC}/${total_checks} verificaciones pasadas"
echo ""

if [ $passed_checks -eq $total_checks ]; then
    echo -e "${GREEN}✅ ¡TODO LISTO!${NC} Lumina está correctamente instalado."
    echo ""
    echo "Próximos pasos:"
    echo "1. cd backend && python app.py"
    echo "2. cd frontend && npm start"
    echo "3. Visita http://localhost:3000"
    echo ""
    echo "Para probar las nuevas funcionalidades:"
    echo "python test_new_features.py"
elif [ $passed_checks -ge $((total_checks * 8 / 10)) ]; then
    echo -e "${YELLOW}⚠️  CASI LISTO${NC} - Algunas verificaciones fallaron"
    echo ""
    echo "Revisa los elementos marcados con ❌ arriba."
    echo "Consulta IMPLEMENTACION_COMPLETADA.md para más detalles."
else
    echo -e "${RED}❌ INSTALACIÓN INCOMPLETA${NC}"
    echo ""
    echo "Varios componentes faltan. Sigue estos pasos:"
    echo "1. Lee IMPLEMENTACION_COMPLETADA.md"
    echo "2. Instala dependencias: pip install -r backend/requirements.txt"
    echo "3. Instala frontend: cd frontend && npm install"
fi

echo ""
