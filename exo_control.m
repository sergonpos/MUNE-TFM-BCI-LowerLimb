%% Configurar y arrancar el exo
channel= canChannel('PEAK-System', 'PCAN_USBBUS1');
configBusSpeed(channel, 1000000);
start(channel);
robot= exoH3(channel);
robot.configController.applyConfig();
robot.taskController.setSpeed(10);

%% CONECTAR AL SOCKET TCP DE PYTHON

% Crear cliente TCP
t = tcpclient('localhost', 9999);

disp('Conectado a Python - esperando predicciones...');

%% BUCLE DE TIEMPO REAL
robotBusy = false;  % Variable de control → el robot está disponible

while true
    % Leer una línea del socket → Python envía '0\n' o '1\n'
    data = readline(t);
    prediction = str2double(data);
    
    % Mostrar la predicción
    disp(['Predicción recibida: ', num2str(prediction)]);

     % Controlar el exoesqueleto según la predicción
    if prediction == 1 && (robotBusy == false)
        % Si recibimos "1" y el robot no está ocupado → le decimos que avance
        robotBusy = true; % Bloqueamos nuevas órdenes hasta que termine

        disp('Ejecutando paso...');
        robot.taskController.singleStepRight;
        pause(2)
        %robot.taskController.singleStepLeft();
        %pause(2)
        robot.taskController.stopWalking();
        %pause(1)
        robotBusy = false;  % desbloquear → ya se puede leer otra orden

    elseif prediction == 0 && (robotBusy == false)
        disp('Parando robot...');
        robot.taskController.stopWalking();

    elseif robotBusy
        disp('Robot ocupado, ignorando predicción.');
    end
end

%% Cuando quieras parar manualmente → pulsa Ctrl+C en Matlab
% Y luego para el CAN:
stop(channel);