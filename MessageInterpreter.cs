// MessageInterpreter
//     > (Last edit)
//     > 29/09/2021
//     > Author: Víctor Martínez-Cagigal

using System.Collections;
using System.Collections.Generic;
using System;
using UnityEngine;
using Newtonsoft.Json;

public class MessageInterpreter
{
    public MessageInterpreter()
    {
    }

    public string decodeEventType(string message)
    {
        return EventTypeDecoder.getEventTypeFromJSON(message);
    }

    public ParameterDecoder decodeParameters(string message)
    {
        return ParameterDecoder.getParametersFromJSON(message);
    }

    public string decodeException(string message)
    {
        return ExceptionDecoder.getExceptionFromJSON(message);
    }

    public int decodeSamples(string message)
    {
        return DecodeSamples.getNoSamplesFromJSON(message);
    }

    public float decodeClassificationValue(string message)
    {
        return ClassificationDecoder.getClassificationValueFromJSON(message);
    }

    public class ClassificationDecoder
    {
        public float value;

        public static float getClassificationValueFromJSON(string jsonString)
        {
            ClassificationDecoder data = JsonUtility.FromJson<ClassificationDecoder>(jsonString);
            return data.value;
        }
    }


    public class EventTypeDecoder
    {
        public string event_type;

        public static string getEventTypeFromJSON(string jsonString)
        {
            EventTypeDecoder event_type = JsonUtility.FromJson<EventTypeDecoder>(jsonString);
            return event_type.event_type;
        }
    }

    public class ParameterDecoder
    {
        public List<int> trialSequence;

        public static ParameterDecoder getParametersFromJSON(string jsonString)
        {
            ParameterDecoder p = JsonUtility.FromJson<ParameterDecoder>(jsonString);
            return p;
        }
    }

    public class ExceptionDecoder
    {
        public string exception;

        public static string getExceptionFromJSON(string jsonString)
        {
            ExceptionDecoder exception = JsonUtility.FromJson<ExceptionDecoder>(jsonString);
            return exception.exception;
        }
    }

    // This function allows us to decode the number of samples received from MEDUSA
    public class DecodeSamples
    {
        public int no_samples;

        public static int getNoSamplesFromJSON(string jsonString)
        {
            DecodeSamples samples = JsonUtility.FromJson<DecodeSamples>(jsonString);
            return samples.no_samples;
        }
    }
}

public class ServerMessage
{
    public Dictionary<string, object> message = new Dictionary<string, object>();

    public ServerMessage(string action)
    {
        message.Add("event_type", action);
    }

    public void addValue(string key, object value)
    {
        message.Add(key, value);
    }

    public string ToJson()
    {
        return JsonConvert.SerializeObject(message);
    }
}

