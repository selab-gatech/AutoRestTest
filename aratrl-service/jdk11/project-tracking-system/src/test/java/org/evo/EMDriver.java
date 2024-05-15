package org.evo;


import org.evomaster.client.java.controller.EmbeddedSutController;
import org.evomaster.client.java.controller.InstrumentedSutStarter;
import org.evomaster.client.java.controller.problem.ProblemInfo;
import org.evomaster.client.java.controller.problem.RestProblem;
import org.evomaster.client.java.controller.api.dto.AuthenticationDto;
import org.evomaster.client.java.controller.api.dto.SutInfoDto;
import org.springframework.boot.SpringApplication;
import org.springframework.context.ConfigurableApplicationContext;

import java.sql.Connection;
import java.util.List;
import java.util.Map;

/**
 * Class used to start/stop the SUT. This will be controller by the EvoMaster process
 */
public class EMDriver extends EmbeddedSutController {

    public static void main(String[] args) {

        int port = 40119;
        if (args.length > 0) {
            port = Integer.parseInt(args[0]);
        }

        EMDriver controller = new EMDriver(40119);
        InstrumentedSutStarter starter = new InstrumentedSutStarter(controller);

        starter.start();
    }


    private ConfigurableApplicationContext ctx;


    public EMDriver() {
        this(40119);
    }

    public EMDriver(int port) {
        setControllerPort(port);
    }

    @Override
    public String startSut() {

        ctx = SpringApplication.run(com.pfa.pack.ProjectTrackingSystemApplication.class, new String[]{
                "--server.port=0"
        });


        return "http://localhost:" + getSutPort();
    }

    protected int getSutPort() {
        return (Integer) ((Map) ctx.getEnvironment()
                .getPropertySources().get("server.ports").getSource())
                .get("local.server.port");
    }


    @Override
    public boolean isSutRunning() {
        return ctx != null && ctx.isRunning();
    }

    @Override
    public void stopSut() {
        ctx.stop();
    }

    @Override
    public String getPackagePrefixesToCover() {
        return "com.pfa.pack.";
    }

    @Override
    public void resetStateOfSUT() {
    }

    @Override
    public List<AuthenticationDto> getInfoForAuthentication() {
        return null;
    }

    public Connection getConnection() {
        return null;
    }

    @Override
    public String getDatabaseDriverName() {
        return null;
    }

    @Override
    public ProblemInfo getProblemInfo() {
        return new RestProblem(
                "https://raw.githubusercontent.com/codingsoo/test/main/project.yaml",
                null
        );
    }

    @Override
    public SutInfoDto.OutputFormat getPreferredOutputFormat() {
        return SutInfoDto.OutputFormat.JAVA_JUNIT_4;
    }

}
