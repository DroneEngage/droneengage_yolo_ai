#include <stdio.h>
#include <signal.h>
#include <iostream>

#include "./helpers/colors.hpp"
#include "./helpers/helpers.hpp"
#include "./helpers/getopt_cpp.hpp"
#include "./helpers/util_rpi.hpp"
#include "version.hpp"
#include "defines.hpp"
#include "./de_common/messages.hpp"
#include "./de_common/configFile.hpp"
#include "./de_common/localConfigFile.hpp"
#include "./de_common/udpClient.hpp"
#include "./de_common/de_module.hpp"
#include "yolo_ai/yolo_ai_main.hpp"
#include "yolo_ai/yolo_ai_parser.hpp"


#include "./helpers/json_nlohmann.hpp"

using Json_de = nlohmann::json;

#define MESSAGE_FILTER {TYPE_AndruavMessage_AI_Recognition_ACTION,\
                        TYPE_AndruavMessage_AI_Recognition_STATUS\
                        }

// This is a timestamp used as instance unique number. if changed then communicator module knows module has restarted.
std::time_t instance_time_stamp;

de::comm::CModule& cModule= de::comm::CModule::getInstance();

std::time_t time_stamp;

bool exit_me = false;

// UAVOS Current PartyID read from communicator
std::string  PartyID;
// UAVOS Current GroupID read from communicator
std::string  GroupID;
std::string  ModuleKey;
int AndruavServerConnectionStatus = SOCKET_STATUS_FREASH;

de::yolo_ai::CYOLOAI_Main& cYOLOAI_Main = de::yolo_ai::CYOLOAI_Main::getInstance();
de::yolo_ai::CYOLOAI_Parser cYOLOAI_Parser = de::yolo_ai::CYOLOAI_Parser();

de::CConfigFile& cConfigFile = de::CConfigFile::getInstance();
de::CLocalConfigFile& cLocalConfigFile = de::CLocalConfigFile::getInstance();

void quit_handler( int sig );

/**
 * @brief true when exit status.
 * 
 */
bool m_exit = false;

/**
 * @brief hardware serial number
 * 
 */
static std::string hardware_serial;


/**
 * @brief display version info
 * 
 */
void _versionOnly (void)
{
    std::cout << version_string << std::endl;
}


/**
 * @brief display version info
 * 
 */
void _version (void)
{
    std::cout << std::endl << _SUCCESS_CONSOLE_BOLD_TEXT_ "Drone-Engage Visual TOLO-AI Module " << _INFO_CONSOLE_TEXT << "version " << version_string << _NORMAL_CONSOLE_TEXT_ << std::endl;
    
    #ifdef TEST_MODE_NO_HAILO_LINK
    std::cout << std::endl << _ERROR_CONSOLE_BOLD_TEXT_ "NO HAILO ACCESS --- VIRTUAL CALLS ONLY " << _NORMAL_CONSOLE_TEXT_ << std::endl;
    #else
    std::cout << std::endl << _SUCCESS_CONSOLE_BOLD_TEXT_ "HAILO LINKED " << _NORMAL_CONSOLE_TEXT_ << std::endl;
    #endif
}


/**
 * @brief display hardware serial number.
 * 
 */
void _displaySerial (void)
{
    _version ();
    std::cout << std::endl << _INFO_CONSOLE_TEXT "Serial Number: " << _TEXT_BOLD_HIGHTLITED_ << hardware_serial << _NORMAL_CONSOLE_TEXT_ << std::endl;
    
}


/**
 * @brief display help for -h command argument.
 * 
 */
void _usage(void)
{
   _version ();
    std::cout << std::endl << _INFO_CONSOLE_TEXT "Options" << _NORMAL_CONSOLE_TEXT_ << std::endl;
    std::cout << std::endl << _INFO_CONSOLE_TEXT "\t--serial:          display serial number needed for registration" << _NORMAL_CONSOLE_TEXT_ << std::ends;
    std::cout << std::endl << _INFO_CONSOLE_TEXT "\t                   -s " << _NORMAL_CONSOLE_TEXT_ << std::ends;
    std::cout << std::endl << _INFO_CONSOLE_TEXT "\t--config:          name and path of configuration file. default [" << configName << "]" << _NORMAL_CONSOLE_TEXT_ << std::ends;
    std::cout << std::endl << _INFO_CONSOLE_TEXT "\t                   -c ./config.json" << _NORMAL_CONSOLE_TEXT_ << std::ends;
    std::cout << std::endl << _INFO_CONSOLE_TEXT "\t--bconfig:          name and path of configuration file. default [" << localConfigName << "]" << _NORMAL_CONSOLE_TEXT_ << std::ends;
    std::cout << std::endl << _INFO_CONSOLE_TEXT "\t                   -b ./config.local" << _NORMAL_CONSOLE_TEXT_ << std::ends;
    std::cout << std::endl << _INFO_CONSOLE_TEXT "\t--version:         -v" << _NORMAL_CONSOLE_TEXT_ << std::endl;
}





void onReceive (const char * message, int len, Json_de jMsg)
{
    #ifdef DEBUG        
        std::cout << _INFO_CONSOLE_TEXT << "RX MSG: " << message << _NORMAL_CONSOLE_TEXT_ << std::endl;
    #endif
    
    
    if (std::strcmp(jMsg[INTERMODULE_ROUTING_TYPE].get<std::string>().c_str(),CMD_TYPE_INTERMODULE)==0)
    {
        const Json_de cmd = jMsg[ANDRUAV_PROTOCOL_MESSAGE_CMD];
        const int messageType = jMsg[ANDRUAV_PROTOCOL_MESSAGE_TYPE].get<int>();
    
        if (messageType== TYPE_AndruavModule_ID)
        {
            
            const int status = cmd ["g"].get<int>();
            if (AndruavServerConnectionStatus != status)
            {
                //_onConnectionStatusChanged (status);
            }
            AndruavServerConnectionStatus = status;
            
            return ;
        }
    }
    
    cYOLOAI_Parser.parseMessage(jMsg, message, len);
    
}

void initSerial()
{
    helpers::CUtil_Rpi::getInstance().get_cpu_serial(hardware_serial);
    hardware_serial.append(get_linux_machine_id());
}

void initArguments (int argc, char *argv[])
{
    int opt;
    const struct GetOptLong::option options[] = {
        {"config",         true,   0, 'c'},
        {"bconfig",        true,   0, 'b'},
        {"serial",         false,  0, 's'},
        {"version",        false,  0, 'v'},
        {"versiononly",    false,  0, 'o'},
        {"help",           false,  0, 'h'},
        {0, false, 0, 0}
    };
    // adding ':' means there is extra parameter needed
    GetOptLong gopt(argc, argv, "c:vh",
                    options);

    /*
      parse command line options
     */
    while ((opt = gopt.getoption()) != -1) {
        switch (opt) {
        case 'c':
            configName = gopt.optarg;
            break;
        case 'b':
            localConfigName = gopt.optarg;
            break;
        case 'v':
            _version();
            exit(0);
            break;
        case 'o':
            _versionOnly();
            exit(0);
        case 's':
            _displaySerial();
            exit(0);
            break;
        case 'h':
            _usage();
            exit(0);
        default:
            printf("Unknown option '%c'\n", (char)opt);
            exit(1);
        }
    }
}


void initDEModule(int argc, char *argv[])
{
    const Json_de& jsonConfig = cConfigFile.GetConfigJSON();
    de::CLocalConfigFile& cLocalConfigFile = de::CLocalConfigFile::getInstance();
        
    cModule.defineModule(
        MODULE_CLASS_A_RECOGNITION,
        jsonConfig["module_id"],
        cLocalConfigFile.getStringField("module_key"),
        version_string,
        Json_de::array(MESSAGE_FILTER)
    );

    cModule.addModuleFeatures(MODULE_FEATURE_AI_RECOGNITION);
    cModule.setHardware(hardware_serial, ENUM_HARDWARE_TYPE::HARDWARE_TYPE_CPU);
    cModule.setMessageOnReceive (&onReceive);

    int udp_chunk_size = DEFAULT_UDP_DATABUS_PACKET_SIZE;
    
    if (validateField(jsonConfig, "s2s_udp_packet_size",Json_de::value_t::string)) 
    {
        udp_chunk_size = std::stoi(jsonConfig["s2s_udp_packet_size"].get<std::string>());
    }
    else
    {
        std::cout << _LOG_CONSOLE_BOLD_TEXT << "WARNING:" << _INFO_CONSOLE_TEXT << " MISSING FIELD " << _ERROR_CONSOLE_BOLD_TEXT_ << "s2s_udp_packet_size " <<  _INFO_CONSOLE_TEXT << "is missing in config file. default value " << _ERROR_CONSOLE_BOLD_TEXT_  << std::to_string(DEFAULT_UDP_DATABUS_PACKET_SIZE) <<  _INFO_CONSOLE_TEXT <<  " is used." << _NORMAL_CONSOLE_TEXT_ << std::endl;    
    }

    // UDP Server
    cModule.init(jsonConfig["s2s_udp_target_ip"].get<std::string>(),
            std::stoi(jsonConfig["s2s_udp_target_port"].get<std::string>().c_str()),
            jsonConfig["s2s_udp_listening_ip"].get<std::string>() ,
            std::stoi(jsonConfig["s2s_udp_listening_port"].get<std::string>().c_str()),
            udp_chunk_size);

}


void init (int argc, char *argv[]) 
{
	signal(SIGINT,quit_handler);
    signal(SIGTERM,quit_handler);
    
    instance_time_stamp = std::time(nullptr);

    // 1- initialize module
    initArguments (argc, argv);

    // 2- initialize serial
    initSerial();

    // Reading Configuration
    std::cout << std::endl << _SUCCESS_CONSOLE_BOLD_TEXT_ << "=================== " << "STARTING PLUGIN ===================" << _NORMAL_CONSOLE_TEXT_ << std::endl;
    _version();

    std::cout << _LOG_CONSOLE_BOLD_TEXT << std::asctime(std::localtime(&instance_time_stamp)) << instance_time_stamp << _INFO_CONSOLE_BOLD_TEXT<< " seconds since the Epoch" << _NORMAL_CONSOLE_TEXT_ << std::endl;
    

    // Define module features
    
    cConfigFile.initConfigFile (configName.c_str());
    cLocalConfigFile.InitConfigFile (localConfigName.c_str());

    ModuleKey = cLocalConfigFile.getStringField("module_key");
    if (ModuleKey=="")
    {
        ModuleKey = std::to_string(get_time_usec());
        cLocalConfigFile.addStringField("module_key",ModuleKey.c_str());
        cLocalConfigFile.apply();
    }

    
    cYOLOAI_Main.init();

    // should be last
    initDEModule (argc,argv);
}


void uninit ()
{
    cYOLOAI_Main.uninit();
	m_exit = true;

    // end program here
	exit(0);
}

// ------------------------------------------------------------------------------
//   Quit Signal Handler
// ------------------------------------------------------------------------------
// this function is called when you press Ctrl-C
void quit_handler( int sig )
{
	std::cout << _INFO_CONSOLE_TEXT << std::endl << "TERMINATING AT USER REQUEST" <<  _NORMAL_CONSOLE_TEXT_ << std::endl;
	
	try 
    {
        exit_me = true;
        uninit();
	}
	catch (int error)
    {

    }
}

int main (int argc, char *argv[]) 
{
    #ifdef DDEBUG
        std::cout << _INFO_CONSOLE_BOLD_TEXT << " ========================== DDEBUG ENABLED =========================="   << _NORMAL_CONSOLE_TEXT_ << std::endl;
    #endif
	init(argc, argv);

    while (!exit_me)
    {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}
