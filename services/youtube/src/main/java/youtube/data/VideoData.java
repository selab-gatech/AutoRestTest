package youtube.data;

import youtube.api.model.*;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.time.Duration;
import java.time.Instant;
import java.util.*;

public class VideoData {

    public static List<Video> getVideoData() throws ParseException {
        List<Video> videos = new ArrayList<>();

        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'");
        dateFormat.setTimeZone(TimeZone.getTimeZone("UTC"));

        //### Video 1 ###

        SearchResultSnippet snippetV1 = new SearchResultSnippet();
        snippetV1.setChannelId("UC0DZmkupLYwc0yDsfocLh0A");
        snippetV1.setChannelTitle("Jelly");
        snippetV1.setDescription("Playing GTA 5 In A ZOMBIE APOCALYPSE! If you enjoyed this video, watch more here: ...");
        snippetV1.setLiveBroadcastContent(SearchResultSnippet.LiveBroadcastContent.NONE);
        snippetV1.setPublishedAt(dateFormat.parse("2020-03-26T20:18:39Z"));
        snippetV1.setTitle("Playing GTA 5 In A ZOMBIE APOCALYPSE! (Scary)");

        ThumbnailDetails v1ThumbnailDetails = createThumbnailDetails("YVm37gs92ME", true);
        snippetV1.setThumbnails(v1ThumbnailDetails);

        VideoCategory gaming = new VideoCategory();
        gaming.setId("20");
        gaming.setTitle("Gaming");

        Video v1 = new Video("YVm37gs92ME", snippetV1, "other",  gaming, "en", Duration.ofSeconds(709), Video.VideoDimension.THREE_D, Video.VideoDefinition.HD,false, null, null, true, Video.VideoLicense.YOUTUBE,
                Arrays.asList("/m/02ntfj", "/m/025zzc", "/m/0bzvm2"), 4502905L, 295795L, 4399L, null, false, null, true);

        //### Video 2 ###

        SearchResultSnippet snippetV2 = new SearchResultSnippet();
        snippetV2.setChannelId("UCRpjHHu8ivVWs73uxHlWwFA");
        snippetV2.setChannelTitle("Eurovision Song Contest");
        snippetV2.setDescription("Add or download the song to your own playlist: https://ESC2018.lnk.to/Eurovision2018EY Sing along with the karaoke versions: ...");
        snippetV2.setLiveBroadcastContent(SearchResultSnippet.LiveBroadcastContent.NONE);
        snippetV2.setPublishedAt(dateFormat.parse("2018-03-12T13:00:02Z"));
        snippetV2.setTitle("Eleni Foureira - Fuego - Cyprus - Official Music Video - Eurovision 2018");

        ThumbnailDetails v2ThumbnailDetails = createThumbnailDetails("eDSgs6syrgg", true);
        snippetV2.setThumbnails(v2ThumbnailDetails);

        VideoCategory entertainment = new VideoCategory();
        entertainment.setId("24");
        entertainment.setTitle("Entertainment");

        Video v2 = new Video("eDSgs6syrgg", snippetV2, "other",  entertainment, "en", Duration.ofSeconds(195), Video.VideoDimension.TWO_D, Video.VideoDefinition.HD,false, null, null, true, Video.VideoLicense.YOUTUBE,
                Arrays.asList("/m/064t9", "/m/04rlf"), 20732038L, 178093L, 10335L, "ytAgeRestricted", false, null, false);

        //### Video 3 ###

        SearchResultSnippet snippetV3 = new SearchResultSnippet();
        snippetV3.setChannelId("UCGGhM6XCSJFQ6DTRffnKRIw");
        snippetV3.setChannelTitle("Muse");
        snippetV3.setDescription("Watch the music video for \"Uprising\" now! \"Uprising\" was released as the lead single from Muse's fifth studio album, The Resistance, on September 7, 2009.");
        snippetV3.setLiveBroadcastContent(SearchResultSnippet.LiveBroadcastContent.NONE);
        snippetV3.setPublishedAt(dateFormat.parse("2009-10-09T13:15:12Z"));
        snippetV3.setTitle("Muse - Uprising [Official Video]");

        ThumbnailDetails v3ThumbnailDetails = createThumbnailDetails("w8KQmps-Sog", false);
        snippetV3.setThumbnails(v3ThumbnailDetails);

        VideoCategory music = new VideoCategory();
        music.setId("10");
        music.setTitle("Music");

        Video v3 = new Video("w8KQmps-Sog", snippetV3, "other",  music, "en", Duration.ofSeconds(252), Video.VideoDimension.TWO_D, Video.VideoDefinition.HD,false, null, null, true, Video.VideoLicense.YOUTUBE,
                Arrays.asList("/m/06by7", "/m/04rlf", "/m/05rwpb"), 189555351L, 964621L, 22910L, null, false, null, false);

        //### Video 4 ###

        SearchResultSnippet snippetV4 = new SearchResultSnippet();
        snippetV4.setChannelId("UCeB7JZwcar2fVoK2w2f9OwA");
        snippetV4.setChannelTitle("Real Betis Balompié");
        snippetV4.setDescription("Joaquín y el Real Betis Balompié amplían su vinculación hasta 2021. SUSCRÍBETE a nuestro canal ⚽️https://www.youtube.com/user/realbetisoficial ...");
        snippetV4.setLiveBroadcastContent(SearchResultSnippet.LiveBroadcastContent.NONE);
        snippetV4.setPublishedAt(dateFormat.parse("2019-12-26T11:45:00Z"));
        snippetV4.setTitle("JOAKING: la leyenda renueva un año más con el Real Betis");

        ThumbnailDetails v4ThumbnailDetails = createThumbnailDetails("cU827EbKIWg", true);
        snippetV4.setThumbnails(v4ThumbnailDetails);

        VideoCategory sports = new VideoCategory();
        sports.setId("17");
        sports.setTitle("Sports");

        Video v4 = new Video("cU827EbKIWg", snippetV4, "other",  sports, "es", Duration.ofSeconds(173), Video.VideoDimension.TWO_D, Video.VideoDefinition.HD,true, null, null, true, Video.VideoLicense.YOUTUBE,
                Arrays.asList("/m/02vx4", "/m/06ntj"), 37.3565037, -5.9817521, 111173L, 4139L, 78L, null,false, null, false);

        //### Video 5 ###

        SearchResultSnippet snippetV5 = new SearchResultSnippet();
        snippetV5.setChannelId("UCpAyEKkMwTiq8FQ_Jqr4Ckg");
        snippetV5.setChannelTitle("Alex Gasan");
        snippetV5.setDescription("");
        snippetV5.setLiveBroadcastContent(SearchResultSnippet.LiveBroadcastContent.NONE);
        snippetV5.setPublishedAt(dateFormat.parse("2019-07-24T08:54:40Z"));
        snippetV5.setTitle("ALONSO vs SCHUMACHER || ULTIMAS VUELTAS GP SAN MARINO 2005");

        ThumbnailDetails v5ThumbnailDetails = createThumbnailDetails("WId93QmyzW4", false);
        snippetV5.setThumbnails(v5ThumbnailDetails);

        Video v5 = new Video("WId93QmyzW4", snippetV5, "other",  sports, "es", Duration.ofSeconds(984), Video.VideoDimension.TWO_D, Video.VideoDefinition.HD,false, Collections.singletonList("es"), null, true, Video.VideoLicense.YOUTUBE,
                Arrays.asList("/m/06ntj", "/m/0410tth"), 44.344297, 11.719407, 39901L, 502L, 9L, null, false, null, true);

        //### Video 6 ###

        SearchResultSnippet snippetV6 = new SearchResultSnippet();
        snippetV6.setChannelId("UCbdSYaPD-lr1kW27UJuk8Pw");
        snippetV6.setChannelTitle("QuantumFracture");
        snippetV6.setDescription("Si la termodinámica te parece una pesadilla, deberías ver esto. ¡Todo lo imprescindible en 5 minutos! No te pierdas ningún vídeo: solo tienes que.");
        snippetV6.setLiveBroadcastContent(SearchResultSnippet.LiveBroadcastContent.NONE);
        snippetV6.setPublishedAt(dateFormat.parse("2015-03-12T14:00:31Z"));
        snippetV6.setTitle("Las Leyes de la Termodinámica en 5 Minutos");

        ThumbnailDetails v6ThumbnailDetails = createThumbnailDetails("Bvfn6eUhUAc", false);
        snippetV6.setThumbnails(v6ThumbnailDetails);

        VideoCategory scienceTechnology = new VideoCategory();
        scienceTechnology.setId("28");
        scienceTechnology.setTitle("Science & Technology");

        Video v6 = new Video("Bvfn6eUhUAc", snippetV6, "other",  scienceTechnology, "es", Duration.ofSeconds(305), Video.VideoDimension.TWO_D, Video.VideoDefinition.HD,false, null, null, true, Video.VideoLicense.YOUTUBE,
                Collections.singletonList("/m/01k8wb"), 2865790L, 71204L, 1447L, null, false, null, false);

        //### Video 7 ###

        SearchResultSnippet snippetV7 = new SearchResultSnippet();
        snippetV7.setChannelId("UCbdSYaPD-lr1kW27UJuk8Pw");
        snippetV7.setChannelTitle("QuantumFracture");
        snippetV7.setDescription("Si hay un experimento que revela lo extraño que es el mundo... Es el Experimento de la Doble Rendija. ¿Preparado para conocer la naturaleza cuántica de la ...");
        snippetV7.setLiveBroadcastContent(SearchResultSnippet.LiveBroadcastContent.NONE);
        snippetV7.setPublishedAt(dateFormat.parse("2017-05-18T16:04:17Z"));
        snippetV7.setTitle("Este Experimento te Dejará LOCO | La Doble Rendija");

        ThumbnailDetails v7ThumbnailDetails = createThumbnailDetails("Y9ScxCemsPM", true);
        snippetV7.setThumbnails(v7ThumbnailDetails);

        Video v7 = new Video("Y9ScxCemsPM", snippetV7, "other",  scienceTechnology, "es", Duration.ofSeconds(598), Video.VideoDimension.TWO_D, Video.VideoDefinition.HD,true, null, null, true, Video.VideoLicense.YOUTUBE,
                Collections.singletonList("/m/01k8wb"), 7319059L, 187170L, 5903L, null, false, null, false);

        //### Video 8 ###

        SearchResultSnippet snippetV8 = new SearchResultSnippet();
        snippetV8.setChannelId("UCnGjxxnmYTUxRl85hrjtEBg");
        snippetV8.setChannelTitle("DonOmarVEVO");
        snippetV8.setDescription("Best of Don Omar / Lo mejor de Don Omar: https://goo.gl/J5mJHa\nSubscribe here: https://goo.gl/aHzXZf\n\n\nMusic video by Don Omar performing Danza Kuduro. (C) 2010 Machete Music");
        snippetV8.setLiveBroadcastContent(SearchResultSnippet.LiveBroadcastContent.NONE);
        snippetV8.setPublishedAt(dateFormat.parse("2010-08-11T20:39:10Z"));
        snippetV8.setTitle("Don Omar - Danza Kuduro ft. Lucenzo");

        ThumbnailDetails v8ThumbnailDetails = createThumbnailDetails("7zp1TbLFPp8", false);
        snippetV8.setThumbnails(v8ThumbnailDetails);

        Video v8 = new Video("7zp1TbLFPp8", snippetV8, "other",  music, "en", Duration.ofSeconds(236), Video.VideoDimension.TWO_D, Video.VideoDefinition.SD,false, null, Arrays.asList("AT", "RO", "PL", "IE", "LU", "LT", "VA", "LV", "PT", "NL", "HU", "SE", "IT", "BG", "GB", "DK", "SI", "DE", "FI", "EE", "MT", "FR", "CZ", "SK", "BE", "GR", "CY"), true, Video.VideoLicense.YOUTUBE,
                null, 1107703517L, 3106827L, 175750L, null, false, null, false);

        //### Video 9 ###

        SearchResultSnippet snippetV9 = new SearchResultSnippet();
        snippetV9.setChannelId("UC2b0o80VP4cbfPE_v8LQEOA");
        snippetV9.setChannelTitle("Movie Juice");
        snippetV9.setDescription("With 30 years in the business, come follow the older Dillon brother's successful rise to fame whilst starring alongside many other acting greats");
        snippetV9.setLiveBroadcastContent(SearchResultSnippet.LiveBroadcastContent.NONE);
        snippetV9.setPublishedAt(dateFormat.parse("2010-11-09T12:56:16Z"));
        snippetV9.setTitle("Movie Star Bios - Matt Dillon");

        ThumbnailDetails v9ThumbnailDetails = createThumbnailDetails("-_MgRXuBtc8", false);
        snippetV9.setThumbnails(v9ThumbnailDetails);

        Video v9 = new Video("-_MgRXuBtc8", snippetV9, "episode",  entertainment, "en", Duration.ofSeconds(490), Video.VideoDimension.TWO_D, Video.VideoDefinition.SD,false, null, null, true, Video.VideoLicense.YOUTUBE,
                Arrays.asList("/m/02vxn", "/m/02jjt", "/m/0f2f9"), 6689467L, 742L, 489L, null, false, null, false);

        //### Video 10 ###

        SearchResultSnippet snippetV10 = new SearchResultSnippet();
        snippetV10.setChannelId("UCK8sQmJBp8GCxrOtXWBpyEA");
        snippetV10.setChannelTitle("Google");
        snippetV10.setDescription("Throughout history, when times are challenging, the world goes looking for heroes. And this year, searches for heroes — both superheroes and everyday heroes ...");
        snippetV10.setLiveBroadcastContent(SearchResultSnippet.LiveBroadcastContent.NONE);
        snippetV10.setPublishedAt(dateFormat.parse("2019-12-11T08:01:24Z"));
        snippetV10.setTitle("Google — Year in Search 2019");

        ThumbnailDetails v10ThumbnailDetails = createThumbnailDetails("ZRCdORJiUgU", true);
        snippetV10.setThumbnails(v10ThumbnailDetails);

        Video v10 = new Video("ZRCdORJiUgU", snippetV10, "other",  scienceTechnology, "en", Duration.ofSeconds(1201), Video.VideoDimension.TWO_D, Video.VideoDefinition.HD,true, null, null, true, Video.VideoLicense.CREATIVE_COMMON,
                Arrays.asList("/m/06ntj", "/m/025zzc", "/m/0f2f9"), 152955611L, 445402L, 23701L, null, false, null, true);

        //### Video 11 ###

        SearchResultSnippet snippetV11 = new SearchResultSnippet();
        snippetV11.setChannelId("UCxwvutsrqponmlkjihgfedc");
        snippetV11.setChannelTitle("My channel");
        snippetV11.setDescription("Hi! This is my first video, hope you enjoy it. Don't forget to like and subscribe!");
        snippetV11.setLiveBroadcastContent(SearchResultSnippet.LiveBroadcastContent.NONE);
        snippetV11.setPublishedAt(dateFormat.parse("2020-04-02T16:20:00Z"));
        snippetV11.setTitle("My first video");

        ThumbnailDetails v11ThumbnailDetails = createThumbnailDetails("ABCD3fgH1_j", true);
        snippetV11.setThumbnails(v11ThumbnailDetails);

        VideoCategory videoBlogging = new VideoCategory();
        videoBlogging.setId("21");
        videoBlogging.setTitle("Videoblogging");

        Video v11 = new Video("ABCD3fgH1_j", snippetV11, "other",  videoBlogging, "en", Duration.ofSeconds(271), Video.VideoDimension.TWO_D, Video.VideoDefinition.HD,true, null, null, true, Video.VideoLicense.YOUTUBE,
                Collections.singletonList("/m/098wr"), 2507L, 102L, 4L, null, false, null, false);

        //### Video 12 ###

        SearchResultSnippet snippetV12 = new SearchResultSnippet();
        snippetV12.setChannelId("UCxwvutsrqponmlkjihgfedc");
        snippetV12.setChannelTitle("My channel");
        snippetV12.setDescription("We're live now! Come and enjoy!");
        snippetV12.setLiveBroadcastContent(SearchResultSnippet.LiveBroadcastContent.LIVE);
        snippetV12.setPublishedAt(Date.from(Instant.now().minusSeconds(300)));
        snippetV12.setTitle("My first streaming!");

        ThumbnailDetails v12ThumbnailDetails = createThumbnailDetails("GjsIeZU-SaE", true);
        snippetV12.setThumbnails(v12ThumbnailDetails);

        Video v12 = new Video("GjsIeZU-SaE", snippetV12, "other",  videoBlogging, "en", Duration.ofSeconds(0), Video.VideoDimension.TWO_D, Video.VideoDefinition.HD,false, null, null, true, Video.VideoLicense.YOUTUBE,
                Collections.singletonList("/m/098wr"), 207L, 28L, 1L, null, true, 68L, false);

        videos.add(v1);
        videos.add(v2);
        videos.add(v3);
        videos.add(v4);
        videos.add(v5);
        videos.add(v6);
        videos.add(v7);
        videos.add(v8);
        videos.add(v9);
        videos.add(v10);
        videos.add(v11);
        videos.add(v12);

        return videos;
    }

    private static ThumbnailDetails createThumbnailDetails(String videoId, boolean fullThumbnail) {
        ThumbnailDetails thumbnailDetails = new ThumbnailDetails();
        Thumbnail defaultThumbnail = new Thumbnail();
        String baseUrl = "https://i.ytimg.com/vi/";
        defaultThumbnail.setWidth(120);
        defaultThumbnail.setHeight(90);
        defaultThumbnail.setUrl(baseUrl + videoId + "/default.jpg");
        Thumbnail mediumThumbnail = new Thumbnail();
        mediumThumbnail.setWidth(320);
        mediumThumbnail.setHeight(180);
        mediumThumbnail.setUrl(baseUrl + videoId + "/mqdefault.jpg");
        Thumbnail highThumbnail = new Thumbnail();
        highThumbnail.setWidth(480);
        highThumbnail.setHeight(360);
        highThumbnail.setUrl(baseUrl + videoId + "/hqdefault.jpg");
        thumbnailDetails.setDefaultThumbnail(defaultThumbnail);
        thumbnailDetails.setMedium(mediumThumbnail);
        thumbnailDetails.setHigh(highThumbnail);
        if(fullThumbnail) {
            Thumbnail standardThumbnail = new Thumbnail();
            standardThumbnail.setWidth(640);
            standardThumbnail.setHeight(480);
            standardThumbnail.setUrl(baseUrl + videoId + "/sddefault.jpg");
            Thumbnail maxResThumbnail = new Thumbnail();
            maxResThumbnail.setWidth(1280);
            maxResThumbnail.setHeight(720);
            maxResThumbnail.setUrl(baseUrl + videoId + "/maxresdefault.jpg");
            thumbnailDetails.setStandard(standardThumbnail);
            thumbnailDetails.setMaxres(maxResThumbnail);
        }
        return thumbnailDetails;
    }
}
